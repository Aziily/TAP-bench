# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import json
import os
import hydra
import numpy as np
import torch
from pathlib import Path

from typing import Optional
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from cotracker.datasets.utils import collate_fn
from cotracker.models.evaluation_predictor import EvaluationPredictor

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.models.build_cotracker import build_cotracker


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs"

    # Name of the dataset to be used for the evaluation.
    mode: str = "tapvid_davis_first"
    # The root directory of the dataset.
    data_root: str = "./"

    # Path to the pre-trained model checkpoint to be used for the evaluation.
    # The default value is the path to a specific CoTracker model checkpoint.
    checkpoint: str = "./checkpoints/scaled_online.pth"
    # Proportion of depth fusion
    proportions: list = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: int = 5
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    num_uniformly_sampled_pts: int = 0
    sift_size: int = 0
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    offline_model: bool = False
    window_len: int = 16
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0
    local_extent: int = 50

    v2: bool = False

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )

def eval_cycle(video_path, evaluator, predictor, cfg: DefaultConfig):
    from data_utils import get_sketch_data_path, get_depth_root_from_data_root, get_perturbed_data_path
    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    exp_type, set_type = cfg.mode.split('_')[0], '_'.join(cfg.mode.split('_')[1:])
    if exp_type == 'sketch':
        PATHS = get_sketch_data_path(cfg.data_root)
    elif exp_type == 'perturbed':
        PATHS = get_perturbed_data_path(cfg.data_root)
        
    dataset_type, dataset_root, queried_first = PATHS[set_type]
    
    from mydataset import TapVidDepthDataset
    
    test_dataset = TapVidDepthDataset(
        dataset_type=dataset_type,
        data_root=dataset_root,
        depth_root=get_depth_root_from_data_root(dataset_root) \
            if exp_type == 'sketch' else os.path.join(video_path, "video_depth_anything"),
        proportions=cfg.proportions,
        queried_first=queried_first,
        resize_to=[256, 256]
    ) 

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=curr_collate_fn,
    )

    # Timing and conducting the evaluation
    import time

    start = time.time()
    # log_file = os.path.join(cfg.exp_dir, f"result_eval_whole.json")
    evaluate_result = evaluator.evaluate_sequence(
        predictor, test_dataloader, dataset_name=cfg.mode
    )
    end = time.time()
    print(end - start)

    # Saving the evaluation results to a .json file
    evaluate_result = evaluate_result["avg"]
    print("evaluate_result", evaluate_result)
    evaluate_result["time"] = end - start
    # result_file = os.path.join(cfg.exp_dir, f"result_eval_.json")
    # print(f"Dumping eval results to {result_file}.")
    # with open(result_file, "w") as f:
    #     json.dump(evaluate_result, f)
    
    return evaluate_result

def run_eval(cfg: DefaultConfig):
    """
    The function evaluates CoTracker on a specified benchmark dataset based on a provided configuration.

    Args:
        cfg (DefaultConfig): An instance of DefaultConfig class which includes:
            - exp_dir (str): The directory path for the experiment.
            - mode (str): The name of the dataset to be used.
            - data_root (str): The root directory of the dataset.
            - checkpoint (str): The path to the CoTracker model's checkpoint.
            - single_point (bool): A flag indicating whether to evaluate one ground truth point at a time.
            - n_iters (int): The number of iterative updates for each sliding window.
            - seed (int): The seed for setting the random state for reproducibility.
            - gpu_idx (int): The index of the GPU to be used.
    """
    # Creating the experiment directory if it doesn't exist
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # Saving the experiment configuration to a .yaml file in the experiment directory
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    evaluator = Evaluator(cfg.exp_dir)
    cotracker_model = build_cotracker(
        cfg.checkpoint, offline=cfg.offline_model, window_len=cfg.window_len, v2=cfg.v2
    )

    # Creating the EvaluationPredictor object
    predictor = EvaluationPredictor(
        cotracker_model,
        grid_size=cfg.grid_size,
        local_grid_size=cfg.local_grid_size,
        sift_size=cfg.sift_size,
        single_point=cfg.single_point,
        num_uniformly_sampled_pts=cfg.num_uniformly_sampled_pts,
        n_iters=cfg.n_iters,
        local_extent=cfg.local_extent,
        interp_shape=(384, 512),
    )

    if torch.cuda.is_available():
        print("Using GPU")
        predictor.model = predictor.model.cuda()

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    exp_type, set_type = cfg.mode.split('_')[0], '_'.join(cfg.mode.split('_')[1:])
    
    os.makedirs(cfg.exp_dir, exist_ok=True)
    output_file = os.path.join(cfg.exp_dir, f"evaluation_results.txt")
    
    if exp_type == 'sketch':
        score = eval_cycle(cfg.data_root, evaluator, predictor, cfg)
        
        with open(output_file, "w") as f:
            for key, score in score.items():
                f.write(f"{key}: {score}\n")
    
    elif exp_type == 'realworld':
        curr_collate_fn = collate_fn
        from mydataset import RealWorldDataset
        test_dataset = RealWorldDataset(
            data_root=cfg.data_root,
            proportions=cfg.proportions,
            resize_to=[256, 256]
        ) 

        # Creating the DataLoader object
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=curr_collate_fn,
        )

        # Timing and conducting the evaluation
        import time

        start = time.time()
        # log_file = os.path.join(cfg.exp_dir, f"result_eval_whole.json")
        evaluate_result = evaluator.evaluate_sequence(
            predictor, test_dataloader, dataset_name=cfg.mode
        )
        end = time.time()
        print(end - start)

        # Saving the evaluation results to a .json file
        evaluate_result = evaluate_result["avg"]
        print("evaluate_result", evaluate_result)
        evaluate_result["time"] = end - start
        
        with open(output_file, "w") as f:
            for key, score in evaluate_result.items():
                f.write(f"{key}: {score}\n")
        
    elif exp_type == 'perturbed':
        total_oa = {}
        total_aj = {}
        total_dx = {}
        pert_sev_results = {}  # New dictionary for storing perturbation-severity pairs
        pert_root = os.path.join(cfg.data_root, "perturbations")
        for perturbation in os.listdir(pert_root):
            pert_path = os.path.join(pert_root, perturbation)
            
            for severity in range(1, 6, 2):  # Loop through severity levels
                sev_path = os.path.join(pert_path, f"severity_{severity}")
                print(sev_path)

                # Evaluate for current perturbation-severity pair
                score = eval_cycle(sev_path, evaluator, predictor, cfg)

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


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


@hydra.main(config_path="./", config_name="default_config_eval")
def evaluate(cfg: DefaultConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()
