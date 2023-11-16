#bin/sh

python scripts/data/zoom/prepare_align.py --base_folder /home/kanghyun/MisalignSR/datasets/Zoom-to-Learn/train/train --operation crop align normalize --num 6

python scripts/data/zoom/prepare_equalize.py /home/kanghyun/MisalignSR/datasets/Zoom-to-Learn /home/kanghyun/MisalignSR/datasets/ZOOM

python scripts/data/zoom/prepare_meta.py --scoring 'nmi'

python scripts/data/zoom/extract_subimage.py

