from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import pickle
import cv2
import glob
import os
import io
from PIL import Image

from typing import Mapping

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

class TapVidDavis(torch.utils.data.Dataset):
    def __init__(self, dataset_location='../datasets/tapvid_davis'):

        print('loading TAPVID-DAVIS dataset...')

        input_path = '%s/tapvid_davis.pkl' % dataset_location
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data
        print('found %d videos in %s' % (len(self.data), dataset_location))
        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        valids = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when not occluded
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        valids = valids.transpose(1,0) # S,N

        vis_ok = valids[0] > 0
        trajs = trajs[:,vis_ok]
        valids = valids[:,vis_ok]

        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W, C = rgbs[0].shape
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2) # S,C,H,W
        trajs = torch.from_numpy(trajs) # S,N,2
        valids = torch.from_numpy(valids) # S,N

        sample = {
            'rgbs': rgbs,
            'trajs': trajs,
            'valids': valids,
            'visibs': valids,
        }
        return sample

    def __len__(self):
        return len(self.data)

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

class TapVidDepthDavis(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        depth_root,
        proportions,
        dataset_type="davis",
        queried_first=True,
        image_size=(512, 896),
    ):
        self.dataset_type = dataset_type
        self.proportions = proportions
        self.queried_first = queried_first
        self.image_size = image_size
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
        depth_frames = self.depth_dataset[video_name]
        
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
        
        # rgbs = self.points_dataset[video_index]['video'] # list of H,W,C uint8 images
        # depth = self.depth_dataset[video_name]
        rgbs = frames
        depth = depth_frames

        trajs = self.points_dataset[video_index]['points'] # N,S,2 array
        valids = 1-self.points_dataset[video_index]['occluded'] # N,S array
        # note the annotations are only valid when not occluded
        
        target_occ = self.points_dataset[video_index]['occluded']
        # note the annotations are only valid when not occluded
        target_points = self.points_dataset[video_index]['points'].copy()
        target_points *= np.array(
            [self.image_size[1] - 1, self.image_size[0] - 1]
        )
        
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        valids = valids.transpose(1,0) # S,N

        vis_ok = valids[0] > 0
        trajs = trajs[:,vis_ok]
        valids = valids[:,vis_ok]

        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W, C = rgbs[0].shape
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1
        
        a, b, c = self.proportions 

        rgbs = np.stack([
            a * depth[:, :, :, 0] + (1 - a) * rgbs[:, :, :, 0],  # First channel blend
            b * depth[:, :, :, 0] + (1 - b) * rgbs[:, :, :, 1],  # Second channel blend
            c * depth[:, :, :, 0] + (1 - c) * rgbs[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension

        
        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2) # S,C,H,W
        trajs = torch.from_numpy(trajs) # S,N,2
        valids = torch.from_numpy(valids) # S,N

        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        
        sample = {
            'rgbs': rgbs,
            'trajs': trajs,
            'valids': valids,
            'visibs': valids,
            'query_points': query_points,
        }
        return sample

    def __len__(self):
        if self.dataset_type == "kinetics":
            return 1071
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

def resize_video(video: np.ndarray, output_size: tuple) -> np.ndarray:
    """Resizes a video to the given output size.
    
    Args:
        video: [T, H, W, C] float32 array.
        output_size: (height, width) tuple.
    
    Returns:
        A video with the given output size, same dtype as input.
    """
    T, H, W, C = video.shape
    target_h, target_w = output_size
    resized_video = np.zeros((T, target_h, target_w, C), dtype=video.dtype)

    for t in range(T):
        frame = video[t]
        resized_frame = cv2.resize(frame, (target_w, target_h))  # 注意顺序是 (width, height)
        resized_video[t] = resized_frame

    return resized_video

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
        
        # 如果frame尺寸和depth_frame尺寸不一样，resize到depth_frame尺寸
        if frames.shape[1:3] != depth_frames.shape[1:3]:
            H, W = depth_frames.shape[1:3]
            scale_w = (W - 1) / (frames.shape[2] - 1)
            scale_h = (H - 1) / (frames.shape[1] - 1)
            queries_xyt[:, 0] *= scale_w
            queries_xyt[:, 1] *= scale_h
            tracks_xy[..., 0] *= scale_w
            tracks_xy[..., 1] *= scale_h
            
            frames = resize_video(frames, (H, W))
        
        a, b, c = self.proportions
        frames = np.stack([
            a * depth_frames[:, :, :, 0] + (1 - a) * frames[:, :, :, 0],  # First channel blend
            b * depth_frames[:, :, :, 0] + (1 - b) * frames[:, :, :, 1],  # Second channel blend
            c * depth_frames[:, :, :, 0] + (1 - c) * frames[:, :, :, 2]   # Third channel blend
        ], axis=3)  # Stack along the channel dimension
        
        frames = frames.astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]
        depth_frames = depth_frames.astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]

        # Load and normalize
        converted = load_queries_strided_from_npz(queries_xyt, tracks_xy, visibles, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, 2
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"])[0]).permute(1, 0)  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]

        sample = {
            'rgbs': rgbs,
            'trajs': trajs,
            'valids': visibles,
            'visibs': visibles,
            'query_points': query_points,
        }

        return sample
