# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation dataset creation functions."""

import sys
sys.path.append('../../repos/tapnet')

import csv
import functools
import io
import os
from os import path
import pickle
import random
import cv2
import glob
from typing import Iterable, Mapping, Optional, Tuple, Union

from absl import logging

import mediapy as media
import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
import tensorflow_datasets as tfds

from tapnet.utils import transforms

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, output_size)


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts.

  Within Thresh, Occ.

  Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.
     get_trackwise_metrics: if True, the metrics will be computed for every
       track (rather than every video, which is the default).  This means
       every output tensor will have an extra axis [batch, num_tracks] rather
       than simply (batch).

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  summing_axis = (2,) if get_trackwise_metrics else (1, 2)

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=summing_axis,
  ) / np.sum(evaluation_points, axis=summing_axis)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=summing_axis,
    )
    count_visible_points = np.sum(
        visible & evaluation_points, axis=summing_axis
    )
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=summing_axis
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(
        false_positives & evaluation_points, axis=summing_axis
    )
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics


def latex_table(mean_scalars: Mapping[str, float]) -> str:
  """Generate a latex table for displaying TAP-Vid and PCK metrics."""
  if 'average_jaccard' in mean_scalars:
    latex_fields = [
        'average_jaccard',
        'average_pts_within_thresh',
        'occlusion_accuracy',
        'jaccard_1',
        'jaccard_2',
        'jaccard_4',
        'jaccard_8',
        'jaccard_16',
        'pts_within_1',
        'pts_within_2',
        'pts_within_4',
        'pts_within_8',
        'pts_within_16',
    ]
    header = (
        'AJ & $<\\delta^{x}_{avg}$ & OA & Jac. $\\delta^{0}$ & '
        + 'Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & '
        + 'Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & $<\\delta^{0}$ & '
        + '$<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & '
        + '$<\\delta^{4}$'
    )
  else:
    latex_fields = ['PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5']
    header = ' & '.join(latex_fields)

  body = ' & '.join(
      [f'{float(np.array(mean_scalars[x]*100)):.3}' for x in latex_fields]
  )
  return '\n'.join([header, body])


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, sample queries
  strided every query_stride frames, ignoring points that are not visible
  at the selected frames.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.
    query_stride: When sampling query points, search for un-occluded points
      every query_stride frames and convert each one into a query.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3].  The video
        has floats scaled to the range [-1, 1].
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1].
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1].
      trackgroup: Index of the original track that each query point was
        sampled from.  This is useful for visualization.
  """
  tracks = []
  occs = []
  queries = []
  trackgroups = []
  total = 0
  trackgroup = np.arange(target_occluded.shape[0])
  for i in range(0, target_occluded.shape[1], query_stride):
    mask = target_occluded[:, i] == 0
    query = np.stack(
        [
            i * np.ones(target_occluded.shape[0:1]),
            target_points[:, i, 1],
            target_points[:, i, 0],
        ],
        axis=-1,
    )
    queries.append(query[mask])
    tracks.append(target_points[mask])
    occs.append(target_occluded[mask])
    trackgroups.append(trackgroup[mask])
    total += np.array(np.sum(target_occluded[:, i] == 0))

  return {
      'video': frames[np.newaxis, ...],
      'query_points': np.concatenate(queries, axis=0)[np.newaxis, ...],
      'target_points': np.concatenate(tracks, axis=0)[np.newaxis, ...],
      'occluded': np.concatenate(occs, axis=0)[np.newaxis, ...],
      'trackgroup': np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
  }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, use the first
  visible point in each track as the query.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3]
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1]
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1]
  """

  valid = np.sum(~target_occluded, axis=1) > 0
  target_points = target_points[valid, :]
  target_occluded = target_occluded[valid, :]

  query_points = []
  for i in range(target_points.shape[0]):
    index = np.where(target_occluded[i] == 0)[0][0]
    x, y = target_points[i, index, 0], target_points[i, index, 1]
    query_points.append(np.array([index, y, x]))  # [t, y, x]
  query_points = np.stack(query_points, axis=0)

  return {
      'video': frames[np.newaxis, ...],
      'query_points': query_points[np.newaxis, ...],
      'target_points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np.newaxis, ...],
  }


def create_davis_dataset(
    davis_points_path: str,
    query_mode: str = 'strided',
    full_resolution=False,
    resolution: Optional[Tuple[int, int]] = (256, 256),
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on DAVIS data."""
  pickle_path = davis_points_path

  with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    davis_points_dataset = pickle.load(f)

  if full_resolution:
    ds, _ = tfds.load(
        'davis/full_resolution', split='validation', with_info=True
    )
    to_iterate = tfds.as_numpy(ds)
  else:
    to_iterate = davis_points_dataset.keys()

  for tmp in to_iterate:
    if full_resolution:
      frames = tmp['video']['frames']
      video_name = tmp['metadata']['video_name'].decode()
    else:
      video_name = tmp
      frames = davis_points_dataset[video_name]['video']
      if resolution is not None and resolution != frames.shape[1:3]:
        frames = resize_video(frames, resolution)

    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    target_points = davis_points_dataset[video_name]['points']
    target_occ = davis_points_dataset[video_name]['occluded']
    target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

    if query_mode == 'strided':
      converted = sample_queries_strided(target_occ, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(target_occ, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'davis': converted}

def read_videos(folder_path, video_suffix="*.mp4",type='davis',rgb=False):
    depth_videos = {}
    video_paths = glob.glob(os.path.join(folder_path, video_suffix))
    
    for video_path in video_paths:
        if type == 'kinetics':
            video_name = int(os.path.basename(video_path).replace(video_suffix[5:], "").split("_")[-1])
        else:
            video_name = os.path.basename(video_path).replace(video_suffix[1:], "")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ensure grayscale
            frames.append(gray_frame[..., np.newaxis])  # Add channel dimension
        
        cap.release()
        
        if frames:
            frames = np.array(frames)  # [F, H, W, 1]
            if rgb and frames.shape[-1] == 1:
                frames = np.concatenate([frames] * 3, axis=-1)
            
            depth_videos[video_name] = frames
            
    return depth_videos
  
def create_depth_dataset(
    data_root,
    depth_root,
    proportions,
    dataset_type="davis",
    resolution=[256, 256],
    query_mode="first",
):
    # dataset_type = dataset_type
    # resize_to = resize_to
    # queried_first = queried_first
    # proportions = proportions
    
    if dataset_type == "kinetics":
        all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
        points_dataset = []
        for pickle_path in all_paths:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                points_dataset = points_dataset + data
        points_dataset = points_dataset

    elif dataset_type == "robotap":
        all_paths = glob.glob(os.path.join(data_root, "robotap_split*.pkl"))
        points_dataset = None
        for pickle_path in all_paths:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                if points_dataset is None:
                    points_dataset = dict(data)
                else:
                    points_dataset.update(data)
        points_dataset = points_dataset
        video_names = list(points_dataset.keys())
        
    else:
        with open(data_root, "rb") as f:
            points_dataset = pickle.load(f)

    video_dataset = read_videos(depth_root, "*_src.mp4", rgb=True)
    depth_dataset = read_videos(depth_root, "*_vis.mp4")
        
    if dataset_type == "davis":
        video_names = list(points_dataset.keys())
    elif dataset_type == "stacking":
        # video_names = [f"video_{i:04d}" for i in range(len(points_dataset))]
        video_names = [f"{i:04d}" for i in range(len(points_dataset))]
        
    print("found %d unique videos in %s" % (len(points_dataset), data_root))
    
    to_iterate = range(len(points_dataset))

    for index in to_iterate:
        if dataset_type in ["davis", "stacking", "robotap"]:
            video_name = video_names[index]
        else:
            video_name = index
            
        if dataset_type in ["davis", "robotap"]:
            video_index = video_name
        else:
            video_index = index

        frames = video_dataset[video_name]
        depth_frames = depth_dataset[video_name]
        # frames = points_dataset[video_index]['video']

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])
        if isinstance(depth_frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            depth_frames = np.array([decode(frame) for frame in depth_frames])
        
        if resolution is not None and resolution != frames.shape[1:3]:
            frames = resize_video(frames, resolution)
            depth_frames = resize_video(depth_frames, resolution)

        a, b, c = proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension
        
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = points_dataset[video_index]['points']
        target_occ = points_dataset[video_index]['occluded']
        target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

        if query_mode == 'strided':
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == 'first':
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f'Unknown query mode {query_mode}.')

        yield {'depth': converted}


def load_queries_strided_from_npz(
    queries_xyt: np.ndarray,
    tracks_xy: np.ndarray,
    visibles: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Loads query and track data from pre-sampled strided format in npz.

    Args:
      queries_xyt: [n_queries, 3], with [x, y, t] in pixel coordinates.
      tracks_xy: [n_frames, n_queries, 2], with [x, y] in pixel coordinates.
      visibles: [n_frames, n_queries] boolean indicating visibility.
      frames: [n_frames, H, W, 3] float32 array scaled to [-1, 1].

    Returns:
      A dict similar to the one returned by `sample_queries_strided`.
    """
    n_frames, H, W, _ = frames.shape
    n_queries = queries_xyt.shape[0]

    # Normalize coordinates to [-1, 1]
    query_points = np.stack([
        # (queries_xyt[:, 2] / (n_frames - 1)) * 2 - 1,  # t in [-1, 1]
        # (queries_xyt[:, 1] / (H - 1)) * 2 - 1,         # y in [-1, 1]
        # (queries_xyt[:, 0] / (W - 1)) * 2 - 1,         # x in [-1, 1]
        queries_xyt[:, 2],
        queries_xyt[:, 1],
        queries_xyt[:, 0],
    ], axis=-1)

    norm_tracks = tracks_xy.copy()
    # norm_tracks[..., 0] = (norm_tracks[..., 0] / (W - 1)) * 2 - 1
    # norm_tracks[..., 1] = (norm_tracks[..., 1] / (H - 1)) * 2 - 1
    
    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": np.transpose(norm_tracks, (1, 0, 2))[np.newaxis, ...],
        "occluded": np.logical_not(visibles.T)[np.newaxis, ...],  # [1, N, T]
        "trackgroup": np.arange(n_queries)[np.newaxis, ...],
    }
    
def create_real_dataset(
    data_root,
    proportions,
    resolution=[256, 256],
):
    depth_root = os.path.join(data_root, "video_depth_anything")
    video_paths = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    print(f"Found {len(video_paths)} video files in {data_root}")
    video_dataset = read_videos(depth_root, "*_src.mp4", rgb=True)
    depth_dataset = read_videos(depth_root, "*_vis.mp4")
        
    print("found %d unique videos in %s" % (len(video_paths), data_root))
    
    to_iterate = range(len(video_paths))

    for index in to_iterate:
        path = video_paths[index]
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            images_jpeg_bytes = data["images_jpeg_bytes"]
            queries_xyt = data["queries_xyt"]
            tracks_xy = data["tracks_XY"]
            visibles = data["visibility"]
            # intrinsics_params = data["fx_fy_cx_cy"]  # Optional

        # Decode frames
        frames = np.array([np.array(Image.open(io.BytesIO(b))) for b in images_jpeg_bytes])
        video_name = os.path.splitext(os.path.basename(path))[0]
        # frames = video_dataset[video_name]
        depth_frames = depth_dataset[video_name]

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])
        if isinstance(depth_frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            depth_frames = np.array([decode(frame) for frame in depth_frames])
        
        if resolution is not None and resolution != frames.shape[1:3]:
            H, W = resolution
            # Rescale the tracks and queries
            scale_w = (W - 1) / (frames.shape[2] - 1)
            scale_h = (H - 1) / (frames.shape[1] - 1)
            queries_xyt[:, 0] *= scale_w
            queries_xyt[:, 1] *= scale_h
            tracks_xy[..., 0] *= scale_w
            tracks_xy[..., 1] *= scale_h
            frames = resize_video(frames, resolution)
            depth_frames = resize_video(depth_frames, resolution)

        a, b, c = proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)
        
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        converted = load_queries_strided_from_npz(queries_xyt, tracks_xy, visibles, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        yield {'depth': converted}