#bin/bash

conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
conda activate MisalignSR
python setup.py develop

# For demo training
python scripts/prepare_example_data.py
python misalignSR/train.py -opt options/example_option.yml