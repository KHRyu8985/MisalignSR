from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import argparse
from PIL import Image
import numpy as np
import utils as utils
import PIL
import cv2
import glob
import shutil
import multiprocessing
import click
from functools import partial
from shutil import copy2


def copy_and_rename_files(src_folder, dst_folder, start_index=1, end_index=6):
    # List all subfolders in the source directory
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    for subfolder in subfolders:
        # Extract the folder name (e.g., '00001')
        folder_name = os.path.basename(subfolder)
        # Normalized image path
        normalized_folder = os.path.join(subfolder, 'normalized')
        # List all JPG files in the 'normalized' subfolder
        files = [f for f in os.listdir(normalized_folder) if f.endswith('.JPG')]
        # Sort the files to ensure the correct order
        files.sort()
        # Copy and rename the first image (HR image)
        if files:
            hr_image_path = os.path.join(normalized_folder, files[0])
            hr_dst_path = os.path.join(dst_folder, 'HR', folder_name + '.JPG')
            copy2(hr_image_path, hr_dst_path)
            # Copy and rename the remaining images (LR images)
            for i in range(start_index, min(end_index, len(files))):
                lr_image_path = os.path.join(normalized_folder, files[i])
                lr_dst_folder = os.path.join(dst_folder, 'LR', str(i + 1))
                lr_dst_path = os.path.join(lr_dst_folder, folder_name + '.JPG')
                copy2(lr_image_path, lr_dst_path)

def copy_and_move(original_dataset_path, new_dataset_path):
    # Creating HR and LR directories
    hr_path = os.path.join(new_dataset_path, 'HR')
    lr_path = os.path.join(new_dataset_path, 'LR')

    # Make the directories if they don't already exist
    os.makedirs(hr_path, exist_ok=True)
    for i in range(2, 7):
        os.makedirs(os.path.join(lr_path, str(i)), exist_ok=True)

    # Copy and rename files from the original dataset to the new ZOOM dataset
    copy_and_rename_files(original_dataset_path, new_dataset_path)

    # Return the path to the new ZOOM dataset
    return new_dataset_path

def process_align_folder(path_folder, num):
    folder_name_save = 'aligned'
    print(f"Write to {path_folder}")

    img_hr_filename = f'{1:05d}.JPG'
    img_hr = cv2.imread(os.path.join(path_folder, 'cropped', img_hr_filename))

    img_lr = []
    for i in range(2, num + 1):
        filename = f'{i:05d}.JPG'
        img_path = os.path.join(path_folder, 'cropped', filename)
        img = cv2.imread(img_path)
        img_lr.append(img)

    try:
        im_regs, im_target = utils.local_alignment_ecc(img_lr, img_hr)
    except Exception as e:
        print(f"Error in {path_folder}: {e}")
        shutil.rmtree(path_folder)
        return

    path_folder_save = os.path.join(path_folder, folder_name_save)
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)

    cv2.imwrite(os.path.join(path_folder_save, img_hr_filename), im_target)
    for idx, im_reg in enumerate(im_regs):
        filename = f'{idx+2:05d}.JPG'
        cv2.imwrite(os.path.join(path_folder_save, filename), im_reg)


def process_normalize_folder(path_folder):
    folder_name_save = 'normalized'
    print(f"Write to {path_folder}")

    list_files = glob.glob(os.path.join(path_folder, 'aligned/*.JPG'))
    path_folder_save = os.path.join(path_folder, folder_name_save)

    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)

    # Find the first image
    first_image_path = os.path.join(path_folder, 'aligned/00001.JPG')

    if os.path.exists(first_image_path):
        # Read the first image
        img_first = cv2.imread(first_image_path)

        # Convert to HSV
        img_first_hsv = cv2.cvtColor(img_first, cv2.COLOR_BGR2HSV)

        # Get the luminance
        img_first_lum = img_first_hsv[:, :, 2]

        # Calculate the mean luminance
        mean_lum = np.mean(img_first_lum) * 0.95

        for file in list_files:
            # Read the image
            img = cv2.imread(file)
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Get the luminance
            img_lum = img_hsv[:, :, 2]
            # Normalize the luminance based on the mean of the first image
            img_lum = (img_lum / np.mean(img_lum)) * mean_lum
            img_lum = np.clip(img_lum, 0, 255)
            # Put the luminance back in the image
            img_hsv[:, :, 2] = img_lum
            # Convert to RGB
            img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            # Save the image
            cv2.imwrite(os.path.join(path_folder_save, file.split('/')[-1]), img_bgr)

class args:
    pass

@click.command()
@click.option('--base_folder', '-b', type=click.STRING, prompt='Enter the base folder', default='datasets/Zoom-to-Learn/train/train')
@click.option('--operation', '-o', type=click.STRING, prompt='Enter the operations to perform (separate by comma for multiple)', default='crop,align,normalize',
              help='Process images with specified operation order')
@click.option('--num', default=6, prompt='Enter the number of files per directory',
              help='Number of files per directory', type=int)
@click.option('--output_folder', '-s', type=click.STRING, prompt='Enter the save folder', default='datasets/ZOOM/train')
def main(base_folder, operation, num, output_folder):
    operation = operation.split(',')
    operation = [op.strip() for op in operation] # remove leading and ending spaces
    ARGS = args()
    ARGS.base_folder = base_folder
    ARGS.operation = operation
    ARGS.num = num
    ARGS.output_folder = output_folder

    # Now you can use args similar to how you would use ARGS from argparse
    print(f"Selected base folder: {ARGS.base_folder}")
    print(f"Operations: {ARGS.operation}")
    print(f"Number of files per directory: {ARGS.num}")

    base_folder = ARGS.base_folder
    entries = os.listdir(base_folder)

    folder_list = [os.path.join(base_folder, entry)
                   for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]

    if 'crop' in ARGS.operation:
        Image.MAX_IMAGE_PIXELS = None
        for folder_name in folder_list:
            tname = '00001.JPG'
            isrotate = utils.readOrien_pil(os.path.join(folder_name, tname)) == 3
            template_f = utils.readFocal_pil(os.path.join(folder_name, tname))
            print("Template image has focal length: ", template_f)

            cropped_folder = os.path.join(folder_name, "cropped")
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)

            if isrotate:
                img_rgb = Image.open(os.path.join(folder_name, tname)).rotate(180)
            else:
                img_rgb = Image.open(os.path.join(folder_name, tname))

            img_rgb.save(os.path.join(cropped_folder, "00001.JPG"))

            i = 1
            while i < ARGS.num:
                line = "%05d.JPG" % (i + 1)
                img_f = utils.readFocal_pil(os.path.join(folder_name, line))
                ratio = float(template_f) / float(img_f)
                print("Image %s has focal length: %s " % (os.path.join(folder_name, line), img_f))
                print("Resize by ratio %s" % ratio)

                if isrotate:
                    img_rgb = Image.open(os.path.join(folder_name, line)).rotate(180)
                else:
                    img_rgb = Image.open(os.path.join(folder_name, line))

                cropped = utils.crop_fov(np.array(img_rgb), 1. / ratio)
                cropped = Image.fromarray(cropped)

                img_rgb_s = cropped.resize(
                    (int(cropped.width * ratio), int(cropped.height * ratio)), Image.Resampling.LANCZOS)

                print("Write to %s" % os.path.join(cropped_folder, "%05d.JPG" % (1 + i)))
                img_rgb_s.save(os.path.join(cropped_folder, "%05d.JPG" % (1 + i)), quality=100)
                i += 1

        print('Cropping done')


    if 'align' in ARGS.operation:
        process_align_folder_with_num = partial(process_align_folder, num=ARGS.num)
        with multiprocessing.Pool() as pool:
            pool.map(process_align_folder_with_num, folder_list)
        print('Align done')

    if 'normalize' in ARGS.operation:
        with multiprocessing.Pool() as pool:
            pool.map(process_normalize_folder, folder_list)
        print('Normalize done')

    print('Copying and moving files to new dataset')
    copy_and_move(base_folder, output_folder)
    print(f"Dataset processed and stored at: {output_folder}")

    print('DONE')

if __name__ == "__main__":
    main()
