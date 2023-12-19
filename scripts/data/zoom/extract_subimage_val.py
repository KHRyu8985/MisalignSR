import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from alignformer.utils import scandir


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for ZOOM dataset.

    Args:
        opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and
            longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for ZOOM dataset.

            * ZOOM_train_HR
            * ZOOM_train_LR_bicubic/X2
            * ZOOM_train_LR_bicubic/X3
            * ZOOM_train_LR_bicubic/X4

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    opt['crop_size'] = 500
    opt['step'] = 250
    opt['thresh_size'] = 500
    opt['shift_level'] = 0

    # HR images
    opt['input_folder'] = 'datasets/ZOOM/ZOOM_valid_HR'
    opt['save_folder'] = 'datasets/ZOOM/ZOOM_valid_HR_sub'
    extract_subimages(opt)

    # # LRx2 images
    # opt['input_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X2'
    # opt['save_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X2_sub'
    # extract_subimages(opt)

    # LRx2 real images
    opt['input_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_real/X2'
    opt['save_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_real/X2_sub'
    extract_subimages(opt)

    # # LRx3 images
    # opt['input_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X3'
    # opt['save_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X3_sub'
    # extract_subimages(opt)

    # # LRx4 images
    # opt['input_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X4'
    # opt['save_folder'] = 'datasets/ZOOM/ZOOM_valid_LR_bicubic/X4_sub'
    # extract_subimages(opt)


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
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        return

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """

    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for ZOOM
    img_name = (img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', ''))

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(
        osp.join(opt['save_folder'], f'{img_name}{extension}'), img,
        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
