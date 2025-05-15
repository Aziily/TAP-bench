# RobusTAP Benchmark

## Environment

```bash
git clone --recursive https://github.com/Aziily/TAP-bench.git
cd repos/TAPTR
git checkout v2
cd ../../

# prepare each methods' environment according to the instruction in each repo
# cd repos/co-tracker ...
# cd repos/locotrack ...
# cd repos/pips2 ...
# cd repos/TAPTR ...
```



## Data Prepare

Download the datasets from our hugginface repo: https://huggingface.co/datasets/superviroo3/tap_dataset

Please make sure the datasets are structured as following:

### Sketch Datasets
```
sketch_*/
└── tapvid_*/
    ├── *.pkl
    └── video_depth_anything/
        ├── *_src.mp4
        └── *_vis.mp4
```

### Perturbation Datasets
```
perturbed_tapvid_*/
├── perturbations/
│   └── [PERTURBATION_NAME]/
│       └── severity_[LEVEL]
│           └── video_depth_anything/
│               ├── *_src.mp4
│               └── *_vis.mp4
└── tapvid_*/
    └── tapvid_*.pkl

```

### Real-World Driving Dataset
```
drivetrack_2d/
├── tapvid3d_*.npz
└── video_depth_anything/
    ├── *_src.mp4
    └── *_vis.mp4
```



## Usage

After prepare the environments, just run under the corresponding folder

```bash
cd scripts/[METHOD]
bash run.bash
```

Here are some parameters may need modification.

| Name           | Meaning                             | Example                               |
| -------------- | ----------------------------------- | ------------------------------------- |
| EXPTYPE        | The data type, choose from "sketch", "perturbed", "real"                      | `sketch`                            |
| DATASET        | The dataset name                    | `tapvid_rgb_stacking_first`         |
| DATAROOT       | The dataset root dir                | `./datasets/tap/sketch_tapvid_rgbs` |
| PROPORTIONS    | The mix rate for each depth channel | `"0.1 0.1 0.1"`                       |
| EXP_NAME       | The name you want to save as        | `sketch_rgbs`                       |
| PYTHON_PATH       | The path to python        | `python`                       |
| DEVICE_ID       | The GPU ID you want to choose, default 0        | `1`                       |

Here's a example running command: `` ./run.bash sketch tapvid_rgb_stacking_first datasets/tap/sketch_tapvid_rgbs "0.0 0.0 0.0" taptrv2_base python 3 ``

After running, the experiments' logs will be saved in `logs/{$EXP_NAME}`
