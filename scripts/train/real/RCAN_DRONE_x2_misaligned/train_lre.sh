#bin/sh

python misalignSR/train.py -opt options/train/misaligned/train_RCAN_DRONE_x2_LRE_misaligned_L1.yml
# python misalignSR/train.py -opt options/train/misaligned/train_RCAN_DRONE_x2_LRE_misaligned_L1.yml --auto_resume # when training is interrupted, use this to resume training
