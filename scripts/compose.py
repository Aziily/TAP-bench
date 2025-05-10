import os
import json
import glob
import shutil
import prettytable
import pandas as pd
from natsort import natsorted

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Summarize performance of different methods.")

    parser.add_argument('--save_dir', type=str, default='outputs', help="Directory to save the summary.")
    parser.add_argument('--methods', type=str, nargs='+', help="List of methods to summarize.")
    parser.add_argument('--identifier', type=str, default='default', help="Identifier for the summary.")
    parser.add_argument('--exp_name', type=str, default='default', help="Name of the experiment.")
    parser.add_argument('--exp_type', type=str, default='sketch', help="Set type to summarize.")

    return parser.parse_args()

METHOD_LIST = {
    "cotracker": "cotracker",
    "locotrack": "locotrack",
    "pips2": "pips2",
    "taptr": "taptr",
    "bootstapir": "tapnet",
    "tapir": "tapnet",
    "tapnext": "tapnext",
}

ALL_CASES = [
    "Brightness-severity_1",
    "Brightness-severity_3",
    "Brightness-severity_5",
    "Contrast-severity_1",
    "Contrast-severity_3",
    "Contrast-severity_5",
    "Defocus_Blur-severity_1",
    "Defocus_Blur-severity_3",
    "Defocus_Blur-severity_5",
    "Fog-severity_1",
    "Fog-severity_3",
    "Fog-severity_5",
    "Frost-severity_1",
    "Frost-severity_3",
    "Frost-severity_5",
    "Gaussian_Noise-severity_1",
    "Gaussian_Noise-severity_3",
    "Gaussian_Noise-severity_5",
    "Glass_Blur-severity_1",
    "Glass_Blur-severity_3",
    "Glass_Blur-severity_5",
    "Impulse_Noise-severity_1",
    "Impulse_Noise-severity_3",
    "Impulse_Noise-severity_5",
    "JPEG-severity_1",
    "JPEG-severity_3",
    "JPEG-severity_5",
    "Motion_Blur-severity_1",
    "Motion_Blur-severity_3",
    "Motion_Blur-severity_5",
    "Pixelate-severity_1",
    "Pixelate-severity_3",
    "Pixelate-severity_5",
    "Shot_Noise-severity_1",
    "Shot_Noise-severity_3",
    "Shot_Noise-severity_5",
    "Snow-severity_1",
    "Snow-severity_3",
    "Snow-severity_5",
    "Zoom_Blur-severity_1",
    "Zoom_Blur-severity_3",
    "Zoom_Blur-severity_5",
]

def summarize_one(method, save_dir, exp_name, exp_type):
    
    if method == "cotracker":
        exp_dir = natsorted(glob.glob(f"{METHOD_LIST[method]}/outputs/*/*/logs/{exp_name}"))[-1]
    elif method == "bootstapir" or method == "tapir":   
        exp_dir = os.path.join(f"{METHOD_LIST[method]}", "logs", method, exp_name)
    else:
        exp_dir = os.path.join(f"{METHOD_LIST[method]}", "logs", exp_name)
    assert os.path.exists(exp_dir) and os.path.isdir(exp_dir)
    
    exp_file = os.path.join(exp_dir, "evaluation_results.txt")
    shutil.copyfile(exp_file, os.path.join(save_dir, f"{method}_eval.txt"))
    
    if exp_type == "sketch" or args.exp_type == "realworld":
        
        res = {
            "occlusion_accuracy": None,
            "average_jaccard": None,
            "average_pts_within_thresh": None
        }
        
        try:
            with open(exp_file, "r") as f:
                for line in f.readlines():
                    if "occlusion_accuracy" in line:
                        res["occlusion_accuracy"] = float(line.split(": ")[-1])
                    elif "average_jaccard" in line:
                        res["average_jaccard"] = float(line.split(": ")[-1])
                    elif "average_pts_within_thresh" in line:
                        res["average_pts_within_thresh"] = float(line.split(": ")[-1])
        except Exception as e:
            print(f"[ERROR] error when reading {exp_file}: {e}")
                    
        return res
    
    elif exp_type == "perturbed":

        res = {
            "all-occlusion_accuracy": None,
            "all-average_jaccard": None,
            "all-average_pts_within_thresh": None
        }

        for case in ALL_CASES:
            res.update({
                f"{case}-occlusion_accuracy": None,
                f"{case}-average_jaccard": None,
                f"{case}-average_pts_within_thresh": None
            })
        
        try:
            with open(exp_file, "r") as f:
                for line in f.readlines():
                    if ": " in line:
                        key, value = line.split(": ")
                        if key in res:
                            res[key] = float(value)
        except Exception as e:
            print(f"[ERROR] error when reading {exp_file}: {e}")
                        
        return res

def del_data(data):
    if data is None:
        return '-'
    if isinstance(data, float):
        if data < 1:
            data = data * 100
        return round(data, 2)
    else:
        return str(data)

def summarize(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    assert args.exp_type in ["sketch", "perturbed", "realworld"]
    
    if args.exp_type == "sketch" or args.exp_type == "realworld":
        table = prettytable.PrettyTable()
        table.title = f"{args.identifier} Summary"
        table.field_names = ["Method", "occlusion_accuracy", "average_jaccard", "average_pts_within_thresh"]

        assert all([method in METHOD_LIST.keys() for method in args.methods])
        
        for method in args.methods:
            data = summarize_one(method, args.save_dir, args.exp_name, args.exp_type)
            print(data)
            row = [method, del_data(data["occlusion_accuracy"]), del_data(data["average_jaccard"]), del_data(data["average_pts_within_thresh"])]
            table.add_row(row, divider=True)
        
        print(table)
        with open(f"{args.save_dir}/{args.identifier}_summary.txt", "w") as f:
            f.write(str(table))
            
    elif args.exp_type == "perturbed":
        table = prettytable.PrettyTable()
        table.title = f"{args.identifier} Summary"
        table.field_names = ["Method", *args.methods]
        
        assert all([method in METHOD_LIST.keys() for method in args.methods])
        
        all_data = {
            "all-occlusion_accuracy": [],
            "all-average_jaccard": [],
            "all-average_pts_within_thresh": []
        }

        for case in ALL_CASES:
            all_data.update({
                f"{case}-occlusion_accuracy": [],
                f"{case}-average_jaccard": [],
                f"{case}-average_pts_within_thresh": []
            })
            
        for method in args.methods:
            data = summarize_one(method, args.save_dir, args.exp_name, args.exp_type)
            for key in all_data:
                all_data[key].append(data[key])
                
        for key in all_data:
            row = [key, *all_data[key]]
            table.add_row(row, divider=True)
            
        print(table)
        with open(f"{args.save_dir}/{args.identifier}_summary.txt", "w") as f:
            f.write(str(table))         

if __name__ == '__main__':
    args = parse_args()
    summarize(args)