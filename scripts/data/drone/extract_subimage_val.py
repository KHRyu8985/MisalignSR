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

    It is used for DRONE dataset.

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
        Typically, there are four folders to be processed for DRONE dataset.

            * DRONE_valid_HR
            * DRONE_valid_HR_X3_sub
            * DRONE_valid_LR_real/X50_9

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HRx2 images
    opt['input_folder'] = 'datasets/DRONE/DRONE_valid_HR'
    opt['save_folder'] = 'datasets/DRONE/DRONE_valid_HR_X2'
    opt['resize_size'] = 360
    extract_subimages(opt)

    # HRx3 images
    opt['input_folder'] = 'datasets/DRONE/DRONE_valid_HR'
    opt['save_folder'] = 'datasets/DRONE/DRONE_valid_HR_X3'
    opt['resize_size'] = 540
    extract_subimages(opt)

    # HRx4 images
    opt['input_folder'] = 'datasets/DRONE/DRONE_valid_HR'
    opt['save_folder'] = 'datasets/DRONE/DRONE_valid_HR_X4'
    opt['resize_size'] = 720
    extract_subimages(opt)


    # # LRx50_9 images
    # opt['input_folder'] = 'datasets/DRONE/DRONE_valid_LR_real/X50_9'
    # opt['save_folder'] = 'datasets/DRONE/DRONE_valid_LR_real/X50_9_sub'
    # extract_subimages(opt)


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

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if opt['resize_size'] is not None:
        resized_size = (opt['resize_size'], opt['resize_size'])
        img = cv2.resize(img, resized_size, interpolation=cv2.INTER_CUBIC)  

    cv2.imwrite(
        osp.join(opt['save_folder'], f'{img_name}{extension}'), img,
        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
