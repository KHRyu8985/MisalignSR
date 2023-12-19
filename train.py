# flake8: noqa
import os.path as osp

import misalignSR.archs
import misalignSR.data
import misalignSR.losses
import misalignSR.models
from basicsr.train import train_pipeline
import sys

if __name__ == '__main__':
    option_file = 'options/train/zoom/train_RRDB_x4_MWN_scratch.yml'
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    sys.argv = ['train.py', '-opt', option_file]
    train_pipeline(root_path)