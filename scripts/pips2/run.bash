# EXPTYPE="sketch"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
# EXPTYPE="realworld"
# DATASET="tapvid_realworld_strided"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/drivetrack_2d"
# PROPORTIONS="0.0 0.0 0.0"

# EXP_NAME="pips2_base"
# PYTHON_PATH="python"
# DEVICE_ID=3

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

IMAGE_SIZE="384 512"

CHECKPOINT="/mnt/nas/share/home/zzh/project/TAPs/pips2/reference_model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$DEVICE_ID $PYTHON_PATH eval.py \
    --init_dir $CHECKPOINT \
    --mode  $EXPTYPE'_'$DATASET \
    --data_root $DATAROOT \
    --proportions $PROPORTIONS \
    --image_size $IMAGE_SIZE  \
    --log_dir logs/$EXP_NAME
