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

# prepare_align.py --base_folder datasets/Zoom-to_Learn/train/train --operation crop align normalize --num 6 # If you want to run all the operations
# prepare_align.py --base_folder datasets/Zoom-to_Learn/train/train --operation normalize --num 6 # If you want to run only one operation

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process images with specified operation order.')
parser.add_argument("--base_folder", type=str, default="datasets/Zoom-to_Learn/train/train",
                    required=True, help="root folder that contains the images")
parser.add_argument('--operation', nargs='+', default=['crop', 'align', 'normalize'])
parser.add_argument("--num", default=6, type=int, help="number of files per dir")
ARGS = parser.parse_args()

base_folder = ARGS.base_folder
entries = os.listdir(base_folder)
folder_list = [os.path.join(base_folder, entry) for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]

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

            img_rgb_s = cropped
            print("Write to %s" % os.path.join(cropped_folder, "%05d.JPG" % (1 + i)))
            img_rgb_s.save(os.path.join(cropped_folder, "%05d.JPG" % (1 + i)), quality=100)
            i += 1

    print('Cropping done')

# Perform alignment
if 'align' in ARGS.operation:

    folder_name_save = 'aligned'
    for path_folder in folder_list:
        print(f"Write to {path_folder}")

        img_hr = []
        for i in range(1, ARGS.num):
            filename = f'{i:05d}.JPG'  # Format the filename as 00001.JPG, 00002.JPG, ...
            img_path = os.path.join(path_folder, 'cropped', filename)
            img = cv2.imread(img_path)  # Image to be aligned.
            img_hr.append(img)

        # Reference image (last image).
        img_lr_filename = f'{ARGS.num:05d}.JPG'
        img_lr = cv2.imread(os.path.join(path_folder, 'cropped', img_lr_filename))

        im_regs, im_target = utils.local_alignment(img_hr, img_lr)

        path_folder_save = os.path.join(path_folder, folder_name_save)

        if not os.path.exists(path_folder_save):
            os.makedirs(path_folder_save)

        # Save the reference image
        cv2.imwrite(os.path.join(path_folder_save, img_lr_filename), im_target)

        # Save the aligned images
        for idx, im_reg in enumerate(im_regs):
            filename = f'{idx+1:05d}.JPG'  # Format the filename as 00001.JPG, 00002.JPG, ...
            cv2.imwrite(os.path.join(path_folder_save, filename), im_reg)

    print('Align done')
if 'normalize' in ARGS.operation:

    folder_name_save = 'normalized'

    for path_folder in folder_list:
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
            mean_lum = np.mean(img_first_lum) * 0.8

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
    print('DONE')
