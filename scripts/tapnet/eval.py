import sys
sys.path.append('../../repos/tapnet')

import cv2
import os
import glob
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
import torch
import torch.nn.functional as F

from mydataset import create_depth_dataset, compute_tapvid_metrics
from argparse import ArgumentParser

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
  
def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(
        np.int32
    )  # [num_points, 3]
    return points


def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles

def inference(frames, query_points, model):
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  query_points = query_points.float()
  frames, query_points = frames[None], query_points[None]

  # Model inference
  outputs = model(frames, query_points)
  tracks, occlusions, expected_dist = (
      outputs['tracks'][0],
      outputs['occlusion'][0],
      outputs['expected_dist'][0],
  )

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist)
  return tracks, visibles

def eval_cycle(video_path, model, args):
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data_utils import get_sketch_data_path, get_depth_root_from_data_root, get_perturbed_data_path
    exp_type, set_type = args.mode.split('_')[0], '_'.join(args.mode.split('_')[1:])
    
    if exp_type == 'realworld':
        from mydataset import create_real_dataset
        test_dataset = create_real_dataset(args.data_root, tuple(args.proportions), resolution=tuple(args.image_size))
    else:
        if exp_type == 'sketch':
            PATHS = get_sketch_data_path(args.data_root)
        elif exp_type == 'perturbed':
            PATHS = get_perturbed_data_path(args.data_root)
            
        dataset_type, dataset_root, queried_first = PATHS[set_type]

        test_dataset = create_depth_dataset(
            data_root=dataset_root,
            depth_root=get_depth_root_from_data_root(dataset_root) \
                if exp_type == 'sketch' else os.path.join(video_path, "video_depth_anything"),
            proportions=tuple(args.proportions),
            dataset_type=dataset_type,
            resolution=tuple(args.image_size),
            query_mode='first' if queried_first else 'strided',
        )

    summed_scalars = None
    for sample_idx, sample in enumerate(test_dataset):
        sample = sample['depth']
        frames = np.round((sample['video'][0] + 1) / 2 * 255).astype(np.uint8)
        query_points = sample['query_points'][0]

        frames = torch.tensor(frames).to(device)
        query_points = torch.tensor(query_points).to(device)

        tracks, visibles = inference(frames, query_points, model)

        tracks = tracks.cpu().detach().numpy()
        visibles = visibles.cpu().detach().numpy()
        query_points = query_points.cpu().detach().numpy()
        occluded = ~visibles

        scalars = compute_tapvid_metrics(
            query_points[None],
            sample['occluded'],
            sample['target_points'],
            occluded[None],
            tracks[None],
            query_mode='first',
        )
        scalars = jax.tree.map(lambda x: np.array(np.sum(x, axis=0)), scalars)
        print(sample_idx, scalars)

        if summed_scalars is None:
            summed_scalars = scalars
        else:
            summed_scalars = jax.tree.map(np.add, summed_scalars, scalars)

        num_samples = sample_idx + 1
    mean_scalars = jax.tree.map(lambda x: x / num_samples, summed_scalars)
    # print(mean_scalars)
        
    return mean_scalars

def main(args):
    MODEL_TYPE = 'bootstapir' if 'bootstapir' in args.ckpt_path else 'tapir'
    if MODEL_TYPE == 'bootstapir':  
        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(
            torch.load(args.ckpt_path)
        )
    else:
        model = tapir_model.TAPIR(pyramid_level=0, bilinear_interp_with_depthwise_conv=False, extra_convs=False)
        model.load_state_dict(
            torch.load(args.ckpt_path)
        )

    model = model.to(device)
    model = model.eval()
    torch.set_grad_enabled(False)
    
    exp_type, set_type = args.mode.split('_')[0], '_'.join(args.mode.split('_')[1:])
    
    os.makedirs(args.save_path, exist_ok=True)
    output_file = os.path.join(args.save_path, f"evaluation_results.txt")
    
    if exp_type == 'sketch' or exp_type == 'realworld':
        scores = eval_cycle(args.data_root, model, args)
        
        with open(output_file, "w") as f:
            for key, score in scores.items():
                f.write(f"{key}: {score}\n")
        
    elif exp_type == 'perturbed':
        total_oa = {}
        total_aj = {}
        total_dx = {}
        pert_sev_results = {}  # New dictionary for storing perturbation-severity pairs
        pert_root = os.path.join(args.data_root, "perturbations")
        for perturbation in os.listdir(pert_root):
            pert_path = os.path.join(pert_root, perturbation)
            
            for severity in range(1, 6, 2):  # Loop through severity levels
                sev_path = os.path.join(pert_path, f"severity_{severity}")
                print(sev_path)

                # Evaluate for current perturbation-severity pair
                score = eval_cycle(sev_path, model, args)
                # print(score)
                # Store results for perturbation-severity pair
                key = f"{perturbation}-severity_{severity}"
                pert_sev_results[key] = {
                    'occlusion_accuracy': score['occlusion_accuracy'],
                    'average_jaccard': score['average_jaccard'],
                    'average_pts_within_thresh': score['average_pts_within_thresh']
                }

                # print(f"Processed {key}")

                # Aggregate per perturbation
                total_oa.setdefault(perturbation, []).append(score['occlusion_accuracy'])
                total_aj.setdefault(perturbation, []).append(score['average_jaccard'])
                total_dx.setdefault(perturbation, []).append(score['average_pts_within_thresh'])

        # Compute final per-perturbation averages
        perturbation_avg = {
            perturbation: {
                'occlusion_accuracy': np.mean(total_oa[perturbation]),
                'average_jaccard': np.mean(total_aj[perturbation]),
                'average_pts_within_thresh': np.mean(total_dx[perturbation])
            }
            for perturbation in total_oa
        }

        # Compute final overall averages
        results = {
            'occlusion_accuracy': np.mean(list(total_oa.values())),
            'average_jaccard': np.mean(list(total_aj.values())),
            'average_pts_within_thresh': np.mean(list(total_dx.values()))
        }

        # Save results to a file
        with open(output_file, "w") as f:
            # Summary of all perturbations
            f.write("Summary of all perturbations\n")
            for metric, scores in results.items():
                f.write(f"all-{metric}: {scores}\n")
            f.write("\n")
            
            # Summary of all perturbation-severity pairs
            f.write("Summary of all perturbation-severity pairs\n")
            for perturbation in perturbation_avg.keys():
                # f.write(f"{perturbation}\n")
                for metric, score in perturbation_avg[perturbation].items():
                    f.write(f"{perturbation}-{metric}: {score}\n")
            f.write("\n")
                    
            # Write perturbation-severity pair results
            f.write("Results for each perturbation-severity pair\n")
            for each_perturbation in pert_sev_results.keys():
                for metric, score in pert_sev_results[each_perturbation].items():
                    f.write(f"{each_perturbation}-{metric}: {score}\n")
            f.write("\n")  

  
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="Mode for training or evaluation.")
    parser.add_argument('--data_root', type=str, default='data', help="Root directory for the data.")
    parser.add_argument('--proportions', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="Proportions for train, val, and test datasets.")
    parser.add_argument('--image_size', type=int, default=[256, 256], nargs=2, help="Size of the input images.")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--save_path', type=str, default='snapshots', help="Path to save the logs and checkpoints.")
    return parser.parse_args()
  
if __name__ == '__main__':
    args = parse_args()
    main(args)