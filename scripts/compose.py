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

    return parser.parse_args()

METHOD_LIST = {
    "cotracker": "cotracker",
    "locotrack": "locotrack",
    "pips2": "pips2",
    "taptr": "taptr"
}

def summarize_one(method, save_dir, exp_name):
    res = {
        "occlusion_accuracy": None,
        "average_jaccard": None,
        "average_pts_within_thresh": None
    }
    
    if method == "cotracker":
        # 找到名为exp_name的文件夹，排序选择最后一个
        exp_dir = natsorted(glob.glob(f"{METHOD_LIST[method]}/outputs/*/*/logs/{exp_name}"))[-1]
        shutil.move(exp_dir, f"{save_dir}/cotracker")
        
        with open(f"{save_dir}/cotracker/result_eval_.json", "r") as f:
            data = json.load(f)
            res["occlusion_accuracy"] = data["occlusion_accuracy"]
            res["average_jaccard"] = data["average_jaccard"]
            res["average_pts_within_thresh"] = data["average_pts_within_thresh"]
            
    elif method == "locotrack":
        exp_dir = os.path.join(f"{METHOD_LIST[method]}", "logs", exp_name, "lightning_logs", "version_0")
        shutil.move(exp_dir, f"{save_dir}/locotrack")
        
        # 读取metrics.csv文件中的test/*
        df = pd.read_csv(f"{save_dir}/locotrack/metrics.csv")
        res["occlusion_accuracy"] = df["test/occlusion_accuracy"]
        res["average_jaccard"] = df["test/average_jaccard"]
        res["average_pts_within_thresh"] = df["test/average_pts_within_thresh"]
        
    elif method == "pips2":
        exp_dir = os.path.join(f"{METHOD_LIST[method]}", "logs", exp_name)
        shutil.move(exp_dir, f"{save_dir}/pips2")
        
        with open(f"{save_dir}/pips2/metrics.json", "r") as f:
            data = json.load(f)
            # res["occlusion_accuracy"] = data["occlusion_accuracy"]
            # res["average_jaccard"] = data["average_jaccard"]
            res["average_pts_within_thresh"] = data["d_avg"]
            
    elif method == "taptr":
        exp_dir = os.path.join(f"{METHOD_LIST[method]}", "logs", exp_name)
        shutil.move(exp_dir, f"{save_dir}/taptr")
        
        with open(f"{save_dir}/taptr/metrics.log", "r") as f:
            data = f.readlines()
            for line in data:
                if "occlusion_accuracy" in line:
                    res["occlusion_accuracy"] = float(line.split(": ")[-1])
                elif "average_jaccard" in line:
                    res["average_jaccard"] = float(line.split(": ")[-1])
                elif "average_pts_within_thresh" in line:
                    res["average_pts_within_thresh"] = float(line.split(": ")[-1])
                    
    #将结果转化为浮点，如果小于1则乘以100，最后保留两位小数
    for key in res.keys():
        if res[key] is not None:
            metric_value = float(res[key])
            if metric_value < 1:
                res[key] = round(metric_value * 100, 2)
            else:
                res[key] = round(metric_value, 2)
        else:
            res[key] = "-"
    
    return [method, res["occlusion_accuracy"], res["average_jaccard"], res["average_pts_within_thresh"]]

def summarize(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    table = prettytable.PrettyTable()
    table.title = f"{args.identifier} Summary"
    table.field_names = ["Method", "occlusion_accuracy", "average_jaccard", "average_pts_within_thresh"]

    assert all([method in METHOD_LIST.keys() for method in args.methods])
    
    for method in args.methods:
        row = summarize_one(method, args.save_dir, args.exp_name)
        table.add_row(row)
    
    print(table)
    with open(f"{args.save_dir}/{args.identifier}_summary.txt", "w") as f:
        f.write(str(table))


if __name__ == '__main__':
    args = parse_args()
    summarize(args)