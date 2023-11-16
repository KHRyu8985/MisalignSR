#!/bin/sh

if [ -z "$1" ]; then
    echo "Device ID: 0"
    CUDA_VISIBLE_DEVICES=0 python misalignSR/train.py -opt options/train/misaligned/train_RCAN_x4_ZOOM_LRE_misaligned.yml --auto_resume
else
    echo "Device ID: $1"
    CUDA_VISIBLE_DEVICES=$1 python misalignSR/train.py -opt options/train/misaligned/train_RCAN_x4_ZOOM_LRE_misaligned.yml --auto_resume
fi

