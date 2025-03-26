#!/bin/bash

# 其他参数
RUN_TAPTR=false
CONDA_ENV_PATH="/home/czk/envs"
BASE_PATH="/home/czk/anaconda3/bin"

# 实验参数
SETTYPE="sketch"
DATASET="tapvid_rgb_stacking_first"
DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
PROPORTIONS="0.0 0.0 0.0"

# 保存路径
IDENTIFIER=$SETTYPE'_'$DATASET'_'${PROPORTIONS// /_}
RES_DIR="../res/$IDENTIFIER"
EXECUTE_LOG_DIR="../logs/$IDENTIFIER"
mkdir -p $RES_DIR
mkdir -p $EXECUTE_LOG_DIR

# 实验名称，仅影响方法保存路径
EXP_NAME="sketch_rgbs"

# 方法列表
ENV_PAIRS=(
    # env_name, dir_name, cuda_id
    "cotracker cotracker 0"
    "locotrack locotrack 1"
    "pips2 pips2 2"
    "taptr taptr 3"
)

SPLITER="========================================================================"

METHOD_LIST=""
for env_pair in "${ENV_PAIRS[@]}"; do
    IFS=' ' read -r -a env <<< "$env_pair"
    ENV_NAME=${env[0]}
    DIR_NAME=${env[1]}
    CUDA_ID=${env[2]}

    if [ "$RUN_TAPTR" = false ] && [ "$ENV_NAME" = "taptr" ]; then
        continue
    fi

    echo $SPLITER
    echo "Running Directory: $DIR_NAME with CUDA_ID: $CUDA_ID"
    echo $SPLITER

    METHOD_LIST="$METHOD_LIST $DIR_NAME"

    cd $DIR_NAME

    bash run.bash \
        "$SETTYPE" \
        "$DATASET" \
        "$DATAROOT" \
        "$PROPORTIONS" \
        "$EXP_NAME" \
        "$CONDA_ENV_PATH/$ENV_NAME/bin/python" \
        "$CUDA_ID" > "../$EXECUTE_LOG_DIR/$DIR_NAME.log" 2>&1

    cd ..
done

echo $SPLITER
echo "Finished running all methods, begin to compose the results"
echo $SPLITER

$BASE_PATH/pip install prettytable pandas natsort
$BASE_PATH/python compose.py \
    --save_dir $RES_DIR \
    --methods $METHOD_LIST \
    --identifier $IDENTIFIER \
    --exp_name "$EXP_NAME"