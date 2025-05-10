import sys
sys.path.append('../../repos/tapnet')
import os

import torch
import torchvision
import tqdm

import numpy as np
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
import torch.nn.functional as F
from tapnet import evaluation_datasets

from mydataset import create_depth_dataset, compute_tapvid_metrics
from argparse import ArgumentParser

def run_eval_per_frame(
    model,
    batch,
    get_trackwise_metrics=True,
    radius=8,
    threshold=0.5,
    use_certainty=False,
):
    with torch.no_grad():
        pred_tracks, track_logits, visible_logits, tracking_state = model(
            video=batch['video'][:, :1], query_points=batch['query_points']
        )
        pred_visible = visible_logits > 0
        pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
        pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
            visible_logits.cpu()
        ]
        for frame in tqdm.tqdm(range(1, batch['video'].shape[1])):
        # ***************************************************
        # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
        # ***************************************************
            (
                curr_tracks,
                curr_track_logits,
                curr_visible_logits,
                tracking_state,
            ) = model(
                video=batch['video'][:, frame : frame + 1],
                state=tracking_state,
            )
            curr_visible = curr_visible_logits > 0
            # ***************************************************
            pred_tracks.append(curr_tracks.cpu())
            pred_visible.append(curr_visible.cpu())
            pred_track_logits.append(curr_track_logits.cpu())
            pred_visible_logits.append(curr_visible_logits.cpu())
        tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
        pred_visible = torch.cat(pred_visible, dim=1).transpose(1, 2)
        track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
        visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)

        pred_certainty = tracker_certainty(tracks, track_logits, radius)
        pred_visible_and_certain = (
            F.sigmoid(visible_logits) * pred_certainty
        ) > threshold

        if use_certainty:
            occluded = ~(pred_visible_and_certain.squeeze(-1))
        else:
            occluded = ~(pred_visible.squeeze(-1))

    scalars = evaluation_datasets.compute_tapvid_metrics(
        batch['query_points'].cpu().numpy(),
        batch['occluded'].cpu().numpy(),
        batch['target_points'].cpu().numpy(),
        occluded.numpy() + 0.0,
        tracks.numpy()[..., ::-1],
        query_mode='first',
        get_trackwise_metrics=get_trackwise_metrics,
    )
    return (
        tracks.numpy()[..., ::-1],
        occluded,
        {k: v.sum(0) for k, v in scalars.items()},
    )


# @title Function for raw data to the input format {form-width: "25%"}
def deterministic_eval(cached_dataset, strided=False):
  if not strided:
    for sample in tqdm.tqdm(cached_dataset, disable=True):
      batch = sample['depth'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )

      yield batch
  else:
    for sample in tqdm.tqdm(cached_dataset):
      batch = sample['depth'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      backward_batch = {k: v.copy() for k, v in batch.items()}
      for key in ['visible', 'appearance', 'loss_mask', 'target_points']:
        backward_batch[key] = np.flip(backward_batch[key], axis=2)
      backward_batch['video'] = np.flip(backward_batch['video'], axis=1)
      backward_queries = (
          backward_batch['video'].shape[1]
          - backward_batch['query_points'][..., 0]
          - 1
      )
      backward_batch['query_points'][..., 0] = backward_queries
      yield batch, backward_batch

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
                if exp_type == 'sketch' else os.path.join(args.data_root, "video_depth_anything"),
            proportions=tuple(args.proportions),
            dataset_type=dataset_type,
            resolution=tuple(args.image_size),
            query_mode='first' if queried_first else 'strided',
        )
        
    cached_dataset = []
    for j, batch in enumerate(test_dataset):
        cached_dataset.append(batch)

    standard_eval_scalars_list = []
    preds = []
    for batch in deterministic_eval(cached_dataset):
        batch = {k: torch.from_numpy(v).cuda().float() for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            tracks, occluded, scores = run_eval_per_frame(
                model, batch, get_trackwise_metrics=False, use_certainty=False
            )
        standard_eval_scalars_list.append(scores)
        preds.append((tracks, occluded))
        
    return {
        'average_jaccard': np.mean([
            standard_eval_scalars_list[k]['average_jaccard']
            for k in range(len(standard_eval_scalars_list))
        ]),
        'occlusion_accuracy': np.mean([
            standard_eval_scalars_list[k]['occlusion_accuracy']
            for k in range(len(standard_eval_scalars_list))
        ]),
        'average_pts_within_thresh': np.mean([
            standard_eval_scalars_list[k]['average_pts_within_thresh']
            for k in range(len(standard_eval_scalars_list))
        ]),
    }

    # print('')
    # print(
    #     'AJ',
    #     np.mean([
    #         standard_eval_scalars_list[k]['average_jaccard']
    #         for k in range(len(standard_eval_scalars_list))
    #     ]),
    # )
    # print(
    #     'OA',
    #     np.mean([
    #         standard_eval_scalars_list[k]['occlusion_accuracy']
    #         for k in range(len(standard_eval_scalars_list))
    #     ]),
    # )
    # print(
    #     'PTS',
    #     np.mean([
    #         standard_eval_scalars_list[k]['average_pts_within_thresh']
    #         for k in range(len(standard_eval_scalars_list))
    #     ]),
    # )

def main(args):
    
    model = TAPNext(image_size=(256, 256))
    ckpt_path = args.ckpt_path
    model = restore_model_from_jax_checkpoint(model, ckpt_path)
    model.cuda()

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