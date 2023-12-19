# A. Installation

## 1. Make a conda environment
Python=3.9 버젼을 사용해서 conda environment 생성
``` bash
conda create -n misalignsr python==3.10
conda activate misalignsr
```
## 2. Pytorch install (v.2.1.1-cu118)
```bash
pip install light-the-torch
ltt install torch torchvision torchaudio
```

## 3. torchopt install
```bash
pip install torchopt
```

## 4. CUPY install
```bash
conda install -c conda-forge cupy cuda-version=11.8
```

## 5. Install mmcv
```bash
pip install -U openmim
mim install mmcv
```

## 6. local clone basicsr in a separate directory and install:
``` bash
cd BasicSR
pip install -r requirements.txt
python setup.py develop
```

## 7. Go to local directory (MisalignSR) and install additional stuff
requirement.txt 파일안에 설치 파일들 있음.
```bash
pip install -e .
```