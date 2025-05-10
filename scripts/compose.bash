#!/bin/bash

# 其他参数
RUN_TAPTR=false
CONDA_ENV_PATH="/home/czk/envs"
BASE_PATH="/home/czk/anaconda3/bin"

# 实验参数
# EXPTYPE="sketch"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
# EXPTYPE="perturbed"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/perturbed_tapvid_rgbs"
EXPTYPE="realworld"
DATASET="tapvid_realworld_strided"
DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/drivetrack_2d"
PROPORTIONS="0.0 0.0 0.0"

# 保存路径
IDENTIFIER=$EXPTYPE'_'$DATASET'_'${PROPORTIONS// /_}
RES_DIR="../res/$IDENTIFIER"
EXECUTE_LOG_DIR="../logs/$IDENTIFIER"
mkdir -p $RES_DIR
mkdir -p $EXECUTE_LOG_DIR

# 实验名称，仅影响方法保存路径
# EXP_NAME="sketch_rgbs"
EXP_NAME="realworld"

# 方法列表
ENV_PAIRS=(
    # env_name, dir_name, cuda_id
    "cotracker cotracker 3"
    "locotrack locotrack 3"
    "pips2 pips2 3"
    # "taptr taptr 3"
    "tapnet tapnet 3"
    "tapnet tapnext 3"
)

SPLITER="========================================================================"

METHOD_LIST=""
for env_pair in "${ENV_PAIRS[@]}"; do
    IFS=' ' read -r -a env <<< "$env_pair"
    ENV_NAME=${env[0]}
    DIR_NAME=${env[1]}
    CUDA_ID=${env[2]}

    if [ "$RUN_TAPTR" = false ] && [ "$DIR_NAME" = "taptr" ]; then
        continue
    fi

    echo $SPLITER
    echo "Running Directory: $DIR_NAME with CUDA_ID: $CUDA_ID"
    echo $SPLITER

    if [ "$DIR_NAME" = "tapnet" ]; then
        METHOD_LIST="$METHOD_LIST bootstapir tapir"
    else
        METHOD_LIST="$METHOD_LIST $DIR_NAME"
    fi

done

$BASE_PATH/python compose.py \
    --save_dir $RES_DIR \
    --methods $METHOD_LIST \
    --identifier $IDENTIFIER \
    --exp_name "$EXP_NAME" \
    --exp_type $EXPTYPE