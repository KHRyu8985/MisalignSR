# Introduction to MisalignSR

# A. Get started

## 1. Make a conda environment
Python=3.9 버젼을 사용해서 conda environment 생성
``` bash
conda create -n misalignsr python==3.9
conda activate misalignsr
```

## 2. local clone basicsr in a separate directory and install:
local clone 하는 이유, pip install 시 몇개 함수가 버젼이 안 맞음...
``` bash
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -r requirements.txt
python setup.py develop
```

## 3. Go to local directory (MisalignSR) and install additional stuff
requirement.txt 파일안에 설치 파일들 있음.
```bash
pip install -e .
```

## 4. Additional
precommit library 설치를 해주면 됨 -> git commit 시 training-white line 같은 간단한 에러 보정해줌.
```bash
pre-commit install
```



# B. Preliminaries about basicSR library

basicSR 라이브러리에 대한 간단한 설명임...

Most deep-learning projects can be divided into the following parts:

1. **data**: defines the training/validation data that is fed into,,,,, the model training
2. **arch** (architecture): defines the network structure and the forward steps
3. **model**: defines the necessary components in training (such as loss) and a complete training process (including forward propagation, back-propagation, gradient optimization, *etc*.), as well as other functions, such as validation, *etc*
4. Training pipeline: defines the training process, that is, connect the data-loader, model, validation, saving checkpoints, *etc*

When we are developing a new method, we often improve the **data**, **arch**, and **model**. Most training processes and basic functions are actually shared. Then, we hope to focus on the development of main functions instead of building wheels repeatedly.

Therefore, we have BasicSR, which separates many shared functions. With BasicSR, we just need to care about the development of **data**, **arch**, and **model**.

In order to further facilitate the use of BasicSR, we provide the basicsr package. You can easily install it through `pip install basicsr`. After that, you can use the training process of BasicSR and the functions already developed in BasicSR~

# C. Preparing dataset

## Viewing the images
To view the pair of aligned and misaligned image, use `view_curtain.py` or `view_checkerboard.py` which will generate gradio app
Using the app, user can use image curtain or checkerboard to visualize the misalignment of two images.

Click Flag to save the interesting image pairs!

For viewing the DIV2K dataset, use `voila` interactive app to view lq and gt pairs

```bash
voila --no-browser view_dataset.ipynb
```

## DIV2K

Extract subimage from original DIV2K dataset. This will generate `DIV2K_train_HR_sub` and `DIV2K_valid_LR_sub`. <br>
The code is modified from https://github.com/XPixelGroup/BasicSR/blob/master/scripts/data_preparation/extract_subimages.py <br>
This will generate misaligned dataset also

```bash
python scripts/data/synthetic/extract_subimage.py
```
