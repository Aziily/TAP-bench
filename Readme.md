# TAP Bench Script

## Environment

```bash
git clone --recursive https://github.com/Aziily/TAP-bench.git
cd repos/TAPTR
git checkout v2
cd ../../

# prepare each methods' environment
# cd repos/co-tracker ...
# cd repos/locotrack ...
# cd repos/pips2 ...
# cd repos/TAPTR ...
```



## Data Prepare

**TODO**


## Usage

After prepare the environments, just run it

```bash
cd scripts
bash run.bash
```

Here are some parameters may need modification.

| Name           | Meaning                             | Example                               |
| -------------- | ----------------------------------- | ------------------------------------- |
| RUN_TAPTR      | Whether TAPTR is evaluated          | `false`                             |
| CONDA_ENV_PATH | The conda env root path             | `/home/user/envs`                   |
| BASE_PATH      | Base python bin dir                 | `/home/user/anaconda3/bin`          |
| SETTYPE        | The data type                      | `sketch`                            |
| DATASET        | The dataset name                    | `tapvid_rgb_stacking_first`         |
| DATAROOT       | The dataset root dir                | `./datasets/tap/sketch_tapvid_rgbs` |
| PROPORTIONS    | The mix rate for each depth channel | `0.1 0.1 0.1`                       |
| EXP_NAME       | The name you want to save as        | `sketch_rgbs`                       |

After running, the result will be saved in `res/{$SETTYPE'_'$DATASET'_'${PROPORTIONS// /_}}` and the experiments' logs will be saved in `logs/{$SETTYPE'_'$DATASET'_'${PROPORTIONS// /_}}`
