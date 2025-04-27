# EXPTYPE="sketch"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
# PROPORTIONS="0.0 0.0 0.0"

# EXP_NAME="tapnet_base"
# PYTHON_PATH="python"

if [ $# -lt 6 ]; then
    echo "Usage: bash run.sh EXPTYPE DATASET DATAROOT PROPORTIONS EXP_NAME PYTHON_PATH"
    exit
else
    echo "Running bash run.sh with arguments: $@"
fi

EXPTYPE=$1
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

CHECKPOINT="/mnt/nas/share/home/zzh/project/TAPs/tapnet/checkpoints/bootstapir_checkpoint_v2.pt"

CUDA_VISIBLE_DEVICES=$DEVICE_ID $PYTHON_PATH eval.py \
    --ckpt_path $CHECKPOINT \
    --mode  $EXPTYPE'_'$DATASET \
    --data_root $DATAROOT \
    --proportions $PROPORTIONS \
    --image_size $IMAGE_SIZE \
    --save_path logs/bootstapir/$EXP_NAME

CHECKPOINT="/mnt/nas/share/home/zzh/project/TAPs/tapnet/checkpoints/tapir_checkpoint_panning.pt"

CUDA_VISIBLE_DEVICES=$DEVICE_ID $PYTHON_PATH eval.py \
    --ckpt_path $CHECKPOINT \
    --mode  $EXPTYPE'_'$DATASET \
    --data_root $DATAROOT \
    --proportions $PROPORTIONS \
    --image_size $IMAGE_SIZE \
    --save_path logs/tapir/$EXP_NAME