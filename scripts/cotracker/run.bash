# EXPTYPE="sketch"
# DATASET="tapvid_rgb_stacking_first"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/sketch_tapvid_rgbs"
# EXPTYPE="realworld"
# DATASET="tapvid_realworld_strided"
# DATAROOT="/mnt/nas/share/home/zzh/datasets/tap/drivetrack_2d"
# PROPORTIONS="0.0 0.0 0.0"

# EXP_NAME="cotracker_base"
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

read prop1 prop2 prop3 <<< $PROPORTIONS
IMAGE_SIZE="256 256"

CONFIG="config"
CHECKPOINT="/mnt/nas/share/home/zzh/project/TAPs/co-tracker/checkpoints/scaled_offline.pth"

$PYTHON_PATH eval.py \
    --config-name $CONFIG \
        checkpoint=$CHECKPOINT \
        mode=$EXPTYPE'_'$DATASET \
        data_root=$DATAROOT \
        offline_model=True \
        window_len=60 \
        proportions.0=$prop1 \
        proportions.1=$prop2 \
        proportions.2=$prop3 \
        exp_dir=logs/$EXP_NAME \
        gpu_idx=$DEVICE_ID \