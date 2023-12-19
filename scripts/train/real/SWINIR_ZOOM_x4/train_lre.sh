#!/bin/sh

if [ -z "$1" ]; then
    echo "Device ID: 0"
    CUDA_VISIBLE_DEVICES=0 python misalignSR/train.py -opt options/train/zoom/train_SwinIR_x4_LRE_scratch.yml
else
    echo "Device ID: $1"
    CUDA_VISIBLE_DEVICES=$1 python misalignSR/train.py -opt options/train/zoom/train_SwinIR_x4_LRE_scratch.yml
fi

