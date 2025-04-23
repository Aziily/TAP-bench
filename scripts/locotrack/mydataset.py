import sys
sys.path.append('../../repos/locotrack/locotrack_pytorch')

import csv
import functools
import io
import os
from os import path
import pickle
import random
from typing import Iterable, Mapping, Optional, Tuple, Union

from absl import logging

import mediapy as media
import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import glob

from models.utils import convert_grid_coordinates

import torch
from torch.utils.data import Dataset

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)

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

class CustomDataset(Dataset):
    def __init__(self, data_generator: Iterable[DatasetElement], key: str):
        self.data = list(data_generator)
        self.key = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][self.key]
        data = {k: torch.tensor(v)[0] if isinstance(v, np.ndarray) else v for k, v in data.items()}
        # Convert double to float
        data = {k: v.float() if v.dtype == torch.float64 else v for k, v in data.items()}
        return data

def read_videos(folder_path, video_suffix="*.mp4", rgb=False):
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
        video_names = [f"video_{i:04d}" for i in range(len(points_dataset))]
        
    print("found %d unique videos in %s" % (len(points_dataset), data_root))
    
    to_iterate = range(len(points_dataset))

    for index in to_iterate:
        if dataset_type in ["davis", "stacking"]:
            video_name = video_names[index]
        else:
            video_name = index
            
        if dataset_type == "davis":
            video_index = video_name
        else:
            video_index = index

        frames = video_dataset[video_name]
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

        yield {'davis': converted}
    
def get_eval_dataset(mode, path, video_path, resolution=(256, 256), proportions=[0.0, 0.0, 0.0]):
    datasets = {}
    
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data_utils import get_sketch_data_path, get_depth_root_from_data_root, get_perturbed_data_path
    exp_type, set_type = mode.split('_')[0], '_'.join(mode.split('_')[1:])
    
    if exp_type == 'sketch':
        PATHS = get_sketch_data_path(path)
    elif exp_type == 'perturbed':
        PATHS = get_perturbed_data_path(path)
        
    dataset_type, dataset_root, queried_first = PATHS[set_type]
    
    if exp_type == 'sketch':
        dataset = create_depth_dataset(
            data_root=dataset_root,
            depth_root=get_depth_root_from_data_root(dataset_root),
            proportions=proportions,
            dataset_type=dataset_type,
            resolution=resolution,
            query_mode='first' if queried_first else 'strided',
        )
        datasets[set_type] = CustomDataset(dataset, 'davis')
    elif exp_type == 'perturbed':
        dataset = create_depth_dataset(
            data_root=dataset_root,
            depth_root=os.path.join(video_path, "video_depth_anything"),
            proportions=proportions,
            dataset_type=dataset_type,
            resolution=resolution,
            query_mode='first' if queried_first else 'strided',
        )
        datasets[set_type] = CustomDataset(dataset, 'davis')

    if len(datasets) == 0:
        raise ValueError(f'No dataset found for mode {mode}.')

    return datasets