# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import glob
import torch
import pickle
import numpy as np
import mediapy as media
import random
import cv2
from PIL import Image
from typing import Mapping, Tuple, Union

from cotracker.datasets.utils import CoTrackerData

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


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
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


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
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TapVidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_type="davis",
        resize_to=[256, 256],
        queried_first=True,
        fast_eval=False,
    ):
        local_random = random.Random()
        local_random.seed(42)
        self.fast_eval = fast_eval
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.queried_first = queried_first
        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            if fast_eval:
                points_dataset = local_random.sample(points_dataset, 50)
            self.points_dataset = points_dataset

        elif self.dataset_type == "robotap":
            all_paths = glob.glob(os.path.join(data_root, "robotap_split*.pkl"))
            points_dataset = None
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    if points_dataset is None:
                        points_dataset = dict(data)
                    else:
                        points_dataset.update(data)
            if fast_eval:
                points_dataset_keys = local_random.sample(
                    sorted(points_dataset.keys()), 50
                )
                points_dataset = {k: points_dataset[k] for k in points_dataset_keys}
            self.points_dataset = points_dataset
            self.video_names = list(self.points_dataset.keys())
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            if self.dataset_type == "davis":
                self.video_names = list(self.points_dataset.keys())
            elif self.dataset_type == "stacking":
                # print("self.points_dataset", self.points_dataset)
                self.video_names = [i for i in range(len(self.points_dataset))]
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        if self.dataset_type == "davis" or self.dataset_type == "robotap":
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name]
        frames = video["video"]

        if self.fast_eval and frames.shape[0] > 300:
            return self.__getitem__((index + 1) % self.__len__())
        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"]
        if self.resize_to is not None:
            frames = resize_video(frames, self.resize_to)
            target_points *= np.array(
                [self.resize_to[1] - 1, self.resize_to[0] - 1]
            )  # 1 should be mapped to resize_to-1
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])

        target_occ = self.points_dataset[video_name]["occluded"]
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        return CoTrackerData(
            rgbs,
            trajs,
            visibles,
            seq_name=str(video_name),
            query_points=query_points,
        )

    def __len__(self):
        return len(self.points_dataset)

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

def read_videos2(folder_path, video_suffix="*.mp4", rgb=False):
    depth_videos = {}
    video_paths = glob.glob(os.path.join(folder_path, video_suffix))
    
    for video_path in video_paths:
        video_name = os.path.basename(video_path).replace(video_suffix[1:], "")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure grayscale
            frames.append(gray_frame)  # Add channel dimension
        
        cap.release()
        
        if frames:
            frames = np.array(frames)  # [F, H, W, 1]
            
            depth_videos[video_name] = frames
            
    return depth_videos

class TapVidPerturbedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        depth_root,
        proportions,
        dataset_type="davis",
        resize_to=[256, 256],
        queried_first=True,
        fast_eval=False,
    ):
        local_random = random.Random()
        local_random.seed(42)
        self.fast_eval = fast_eval
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.queried_first = queried_first
        self.proportions = proportions
        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            if fast_eval:
                points_dataset = local_random.sample(points_dataset, 50)
            self.points_dataset = points_dataset

        elif self.dataset_type == "robotap":
            all_paths = glob.glob(os.path.join(data_root, "robotap_split*.pkl"))
            points_dataset = None
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    if points_dataset is None:
                        points_dataset = dict(data)
                    else:
                        points_dataset.update(data)
            if fast_eval:
                points_dataset_keys = local_random.sample(
                    sorted(points_dataset.keys()), 50
                )
                points_dataset = {k: points_dataset[k] for k in points_dataset_keys}
            self.points_dataset = points_dataset
            self.video_names = list(self.points_dataset.keys())
            
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
        
        self.video_dataset = read_videos(depth_root, "*_src.mp4", rgb=True)
        self.depth_dataset = read_videos(depth_root, "*_vis.mp4")
            
        if self.dataset_type == "davis":
            self.video_names = list(self.points_dataset.keys())
        elif self.dataset_type == "stacking":
            self.video_names = [f"{i:04d}" for i in range(len(self.points_dataset))]
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        if self.dataset_type == "davis" or self.dataset_type == "robotap":
            video_name = self.video_names[index]
            video_name_d = video_name + "_vis"
        else:
            video_name = index

        frames = self.video_dataset[video_name]
        depth_frames = self.depth_dataset[video_name]

        if self.fast_eval and frames.shape[0] > 300:
            return self.__getitem__((index + 1) % self.__len__())
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
        target_points = self.points_dataset[video_name]["points"]
        if self.resize_to is not None:
            frames = resize_video(frames, self.resize_to)
            depth_frames = resize_video(depth_frames, self.resize_to)
            target_points *= np.array(
                [self.resize_to[1] - 1, self.resize_to[0] - 1]
            )  # 1 should be mapped to resize_to-1
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])

        target_occ = self.points_dataset[video_name]["occluded"]
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        proportions = self.proportions
        a, b, c = proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension
        
        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        return CoTrackerData(
            rgbs,
            trajs,
            visibles,
            seq_name=str(video_name),
            query_points=query_points,
        )

    def __len__(self):
        return len(self.points_dataset)
    
class TapVidDepthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        depth_root,
        proportions,
        dataset_type="davis",
        resize_to=[256, 256],
        queried_first=True,
        fast_eval=False,
    ):
        local_random = random.Random()
        local_random.seed(42)
        self.fast_eval = fast_eval
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.queried_first = queried_first
        self.proportions = proportions
        self.depth_root = depth_root
        
        if self.dataset_type == "kinetics":
            self.all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            self.curr_path_idx = -1
            self.global_file_idx = 0
            self.points_dataset = [] # initialize to empty list

        elif self.dataset_type == "robotap":
            all_paths = glob.glob(os.path.join(data_root, "robotap_split*.pkl"))
            points_dataset = None
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    if points_dataset is None:
                        points_dataset = dict(data)
                    else:
                        points_dataset.update(data)
            if fast_eval:
                points_dataset_keys = local_random.sample(
                    sorted(points_dataset.keys()), 50
                )
                points_dataset = {k: points_dataset[k] for k in points_dataset_keys}
            self.points_dataset = points_dataset
            self.video_names = list(self.points_dataset.keys())
            
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
        if self.dataset_type != "kinetics":
            self.video_dataset = read_videos(depth_root, "*_src.mp4", rgb=True)
            self.depth_dataset = read_videos(depth_root, "*_vis.mp4")
            
        if self.dataset_type == "davis":
            self.video_names = list(self.points_dataset.keys())
        elif self.dataset_type == "stacking":
            # self.video_names = [f"video_{i:04d}" for i in range(len(self.points_dataset))]
            self.video_names = [f"{i:04d}" for i in range(len(self.points_dataset))]
            
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))
        
    def load_pickle_file(self):
        with open(self.all_paths[self.curr_path_idx], "rb") as f:
            data = pickle.load(f)
        return data

    def __getitem__(self, index):
        if self.dataset_type in ["davis", "stacking", "robotap"]:
            video_name = self.video_names[index]
        else:
            video_name = index
            
        if self.dataset_type in ["davis", "robotap"]:
            video_index = video_name
        else:
            video_index = index
            
        if self.dataset_type == 'kinetics':
            pkl_index = index - self.global_file_idx # index within the current pickle file
            if pkl_index >= len(self.points_dataset):
                self.global_file_idx+=len(self.points_dataset)
                self.curr_path_idx+=1
                print("reading pickle file")
                self.points_dataset = self.load_pickle_file()
                print("reading depth file")
                self.depth_dataset = read_videos(self.depth_root, f"000{self.curr_path_idx}*_vis.mp4", type='kinetics')
                print("reading video file")
                self.video_dataset = read_videos(self.depth_root, f"000{self.curr_path_idx}*_src.mp4", type='kinetics', rgb=True)
                print("reading done!")
                pkl_index = index - self.global_file_idx # index within the new pickle file
            video_name = pkl_index     
            video_index = pkl_index

        frames = self.video_dataset[video_name]
        # frames = self.points_dataset[video_index]['video']
        depth_frames = self.depth_dataset[video_name]

        if self.fast_eval and frames.shape[0] > 300:
            return self.__getitem__((index + 1) % self.__len__())
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
        target_points = self.points_dataset[video_index]["points"]
        if self.resize_to is not None:
            frames = resize_video(frames, self.resize_to)
            depth_frames = resize_video(depth_frames, self.resize_to)
            target_points *= np.array(
                [self.resize_to[1] - 1, self.resize_to[0] - 1]
            )  # 1 should be mapped to resize_to-1
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])

        target_occ = self.points_dataset[video_index]["occluded"]
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        proportions = self.proportions
        a, b, c = proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension
        
        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        return CoTrackerData(
            rgbs,
            trajs,
            visibles,
            seq_name=str(video_name),
            query_points=query_points,
        )

    def __len__(self):
        if self.dataset_type == "kinetics":
            return 1081
        return len(self.points_dataset)
    
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
    

class RealWorldDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, proportions, resize_to=(256, 256)):
        self.data_root = data_root
        depth_root = os.path.join(data_root, "video_depth_anything")
        self.resize_to = resize_to
        self.video_paths = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        print(f"Found {len(self.video_paths)} video files in {data_root}")
        self.video_dataset = read_videos(depth_root, "*_src.mp4", rgb=True)
        self.depth_dataset = read_videos(depth_root, "*_vis.mp4")
        self.proportions = proportions
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        path = self.video_paths[index]
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
        # frames = self.video_dataset[video_name]
        depth_frames = self.depth_dataset[video_name]

        if self.resize_to is not None:
            H, W = self.resize_to
            # Rescale the tracks and queries
            scale_w = (W - 1) / (frames.shape[2] - 1)
            scale_h = (H - 1) / (frames.shape[1] - 1)
            queries_xyt[:, 0] *= scale_w
            queries_xyt[:, 1] *= scale_h
            tracks_xy[..., 0] *= scale_w
            tracks_xy[..., 1] *= scale_h
            frames = resize_video(frames, self.resize_to)
            depth_frames = resize_video(depth_frames, self.resize_to)

        
        a, b, c = self.proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension
        
        # frames = frames.astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]
        # depth_frames = depth_frames.astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]

        # Load and normalize
        converted = load_queries_strided_from_npz(queries_xyt, tracks_xy, visibles, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, 2
        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"])[0]).permute(1, 0)  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]

        return CoTrackerData(
            rgbs,
            trajs,
            visibles,
            seq_name=os.path.basename(path),
            query_points=query_points,
        )
