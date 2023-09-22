# Data preparation

## Viewing the images
To view the pair of aligned and misaligned image, use `view_curtain.py` or `view_checkerboard.py` which will generate gradio app
Using the app, user can use image curtain or checkerboard to visualize the misalignment of two images.

Click Flag to save the interesting image pairs!
## DIV2K

Extract subimage from original DIV2K dataset. This will generate `DIV2K_train_HR_sub` and `DIV2K_valid_LR_sub`. <br>
The code is modified from https://github.com/XPixelGroup/BasicSR/blob/master/scripts/data_preparation/extract_subimages.py <br>
This will generate misaligned dataset also

```bash
python scripts/data/synthetic/extract_subimage.py
```

For viewing the generated dataset, use `voila` interactive app

```bash
voila --no-browser view_dataset.ipynb
```

## Training
