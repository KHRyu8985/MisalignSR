import os.path as osp
import cv2
import glob
import numpy as np
import os
import multiprocessing

def save_img(img, path):
    cv2.imwrite(path, img)

def process_image(img_path, opt):
    print(img_path)
    img_name = osp.splitext(osp.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    idx = 0
    for x in range(0, w - opt['crop_size'] + 1, opt['step']):
        for y in range(0, h - opt['crop_size'] + 1, opt['step']):
            idx += 1
            x_end = min(x + opt['crop_size'], w)
            y_end = min(y + opt['crop_size'], h)
            cropped = img[y:y_end, x:x_end, :]
            if cropped.shape[0] < opt['thresh_size'] or cropped.shape[1] < opt['thresh_size']:
                continue
            save_fn = osp.join(opt['save_folder'], f'{img_name}_s{idx:03d}.png')
            save_img(cropped, save_fn)

def extract_subimages(opt):
    if not osp.exists(opt['save_folder']):
        os.makedirs(opt['save_folder'])

    img_list = sorted(glob.glob(osp.join(opt['input_folder'], '*')))
    with multiprocessing.Pool() as pool:
        pool.starmap(process_image, [(img_path, opt) for img_path in img_list])

'''

# HR images
opt_train = {
    'input_folder': 'datasets/ZOOM/train/HR',
    'save_folder': 'datasets/ZOOM/train/HR_sub',
    'crop_size': 500,
    'step': 250,
    'thresh_size': 500,
    'shift_level': 0
}
extract_subimages(opt_train)

# Scaled images
scales = [2, 3, 4, 5, 6]
for scale in scales:
    folder_scale = str(scale).replace('.', '_')  # Convert 1.5 to 1_5, 2.5 to 2_5, etc.
    opt_train['input_folder'] = f'datasets/ZOOM/train/LR/{folder_scale}'
    opt_train['save_folder'] = f'datasets/ZOOM/train/LR/{folder_scale}_sub'
    extract_subimages(opt_train)
'''

### Same for the test
# HR images
opt_test = {
    'input_folder': 'datasets/ZOOM/test/HR',
    'save_folder': 'datasets/ZOOM/test/HR_sub_large',
    'crop_size': 1200,
    'step': 600,
    'thresh_size': 1200,
}
extract_subimages(opt_test)

# Scaled images
scales = [2, 3, 4, 5, 6]
for scale in scales:
    folder_scale = str(scale).replace('.', '_')  # Convert 1.5 to 1_5, 2.5 to 2_5, etc.
    opt_test['input_folder'] = f'datasets/ZOOM/test/LR/{folder_scale}'
    opt_test['save_folder'] = f'datasets/ZOOM/test/LR/{folder_scale}_sub_large'
    extract_subimages(opt_test)