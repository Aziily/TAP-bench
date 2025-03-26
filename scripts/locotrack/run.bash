# SETTYPE="sketch"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
# PROPORTIONS="0.0 0.0 0.0"

# EXP_NAME="locotrack_base"
# PYTHON_PATH="python"

if [ $# -lt 6 ]; then
    echo "Usage: bash run.sh SETTYPE DATASET DATAROOT PROPORTIONS EXP_NAME PYTHON_PATH"
    exit
else
    echo "Running bash run.sh with arguments: $@"
fi

SETTYPE=$1
DATASET=$2
DATAROOT=$3
PROPORTIONS=$4
EXP_NAME=$5
PYTHON_PATH=$6

if [ $# -eq 7 ]; then
    DEVICE_ID=$7
else
    DEVICE_ID=0
fi

IMAGE_SIZE="256 256"

CONFIG="config.ini"
CHECKPOINT="/mnt/nas/share/home/zzh/project/TAPs/locotrack/checkpoint/locotrack_base.ckpt"

CUDA_VISIBLE_DEVICES=$DEVICE_ID $PYTHON_PATH eval.py \
    --config $CONFIG \
    --ckpt_path $CHECKPOINT \
    --mode  $SETTYPE'_'$DATASET \
    --data_root $DATAROOT \
    --proportions $PROPORTIONS \
    --image_size $IMAGE_SIZE \
    --save_path logs/$EXP_NAME
