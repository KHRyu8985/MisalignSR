import os.path as osp
import cv2
import glob
import numpy as np
import os

def save_img(img, path):
    cv2.imwrite(path, img)

def extract_subimages(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    crop_sz = opt['crop_size']
    step = opt['step']
    thresh_sz = opt['thresh_size']
    shift_level = opt['shift_level']

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    img_list = sorted(glob.glob(osp.join(input_folder, '*')))

    idx = 0
    for img_path in img_list:
        print(img_path)
        img_name = osp.splitext(osp.basename(img_path))[0]  # Removing the '.png' extension
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w, _ = img.shape

        for x in range(0, w - crop_sz + 1, step):
            for y in range(0, h - crop_sz + 1, step):
                idx += 1
                x_end = min(x + crop_sz, w)
                y_end = min(y + crop_sz, h)
                cropped = img[y:y_end, x:x_end, :]
                if cropped.shape[0] < thresh_sz or cropped.shape[1] < thresh_sz:
                    continue
                if shift_level > 0:
                    cropped_shift = np.roll(cropped, shift = shift_level, axis = 0)
                    cropped = np.roll(cropped_shift, shift = shift_level, axis = 1)
                save_fn = osp.join(save_folder, f'{img_name}_s{idx:03d}.png')
                save_img(cropped, save_fn)
# HR images
opt = {
    'input_folder': 'datasets/ZOOM/train/HR',
    'save_folder': 'datasets/ZOOM/train/HR_sub',
    'crop_size': 480,
    'step': 480,
    'thresh_size': 0,
    'shift_level': 0
}
extract_subimages(opt)

# Scaled images
scales = [2, 3, 4, 5]
for scale in scales:
    folder_scale = str(scale).replace('.', '_')  # Convert 1.5 to 1_5, 2.5 to 2_5, etc.
    opt['input_folder'] = f'datasets/ZOOM/train/LR/x{folder_scale}'
    opt['save_folder'] = f'datasets/ZOOM/train/LR/x{folder_scale}_sub'
    opt['crop_size'] = int(480 / scale)
    opt['step'] = int(opt['crop_size'])
    extract_subimages(opt)