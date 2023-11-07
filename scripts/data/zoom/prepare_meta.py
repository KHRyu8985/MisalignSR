from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from defocus.BlurDetector import BlurDetector
from defocus.defocus_estimate import estimate_bmap_laplacian

import cv2
from skimage import filters
import os
from pathlib import Path
import tqdm

def compute_and_save_ncc(hr_patch, lr_patch, hr_meta_path, lr_meta_path, idx):
    """Compute NCC and save the images, return the NCC value."""
    ncc_value = get_ccorr_normed(hr_patch, lr_patch)

    # Save HR and LR patches
    cv2.imwrite(os.path.join(hr_meta_path, f"patch_{idx}.png"), cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(lr_meta_path, f"patch_{idx}.png"), cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))

    return ncc_value

def detect_blur(im):
    image_np = np.asarray(im)
    image_grayscale = np.dot(image_np[...,:3], [0.299, 0.587, 0.114])
    detector = BlurDetector(show_progress=True)
    res, score = detector.detectBlur(image_grayscale)

    val = filters.threshold_otsu(res)
    mask = res > val
    return res, mask, score

def get_top_sharpest_patches_fast(sharpness_map, original_image, patch_size=480, top_n=5):
    height, width = sharpness_map.shape

    # Compute integral image
    integral_image = cv2.integral(sharpness_map)

    sharpness_scores = []

    # Using integral image to get sum in O(1)
    for y in range(height - patch_size):
        for x in range(width - patch_size):
            y2, x2 = y + patch_size, x + patch_size
            current_sharpness = integral_image[y2, x2] - integral_image[y, x2] - integral_image[y2, x] + integral_image[y, x]
            sharpness_scores.append((current_sharpness, (y, x)))

    # Sorting and selecting top patches
    sharpness_scores.sort(reverse=True, key=lambda x: x[0])

    def is_overlapping(box1, box2, patch_size):
        _patch_size = patch_size //2
        y1, x1 = box1
        y2, x2 = box2
        return not (x1 + _patch_size <= x2 or x1 >= x2 + _patch_size or y1 + _patch_size <= y2 or y1 >= y2 + _patch_size)

    selected_patches = []
    selected_positions = []

    for _, position in sharpness_scores:
        if len(selected_patches) == top_n:
            break
        if any(is_overlapping(position, prev_pos, patch_size) for prev_pos in selected_positions):
            continue
        y, x = position
        selected_patches.append(original_image[y:y+patch_size, x:x+patch_size])
        selected_positions.append(position)

    return selected_patches, selected_positions

def crop_lr_patches(lr_image, out_positions, patch_size, scale_factor):
    lr_patches = []
    adjusted_patch_size = int(patch_size / scale_factor)

    for position in out_positions:
        y, x = position
        lr_x = int(x / scale_factor)
        lr_y = int(y / scale_factor)
        lr_patch = lr_image[lr_y:lr_y+adjusted_patch_size, lr_x:lr_x+adjusted_patch_size]
        lr_patches.append(lr_patch)

    return lr_patches

def process_and_save_images(hr_path, lr_path, hr_save_dir, lr_save_dir, scale_factor=4, patch_size=480, top_n=5):
    # Ensure save directories exist
    Path(hr_save_dir).mkdir(parents=True, exist_ok=True)
    Path(lr_save_dir).mkdir(parents=True, exist_ok=True)

    hr_image_names = os.listdir(hr_path)

    for hr_image_name in tqdm.tqdm(hr_image_names, desc="Processing images"):
        hr_image_path = os.path.join(hr_path, hr_image_name)
        lr_image_path = os.path.join(lr_path, hr_image_name)

        hr_image_np = cv2.imread(hr_image_path)
        hr_image_np = cv2.cvtColor(hr_image_np, cv2.COLOR_BGR2RGB)

        lr_image_np = cv2.imread(lr_image_path)
        lr_image_np = cv2.cvtColor(lr_image_np, cv2.COLOR_BGR2RGB)

        sharpness_map, _, _ = detect_blur(hr_image_np)
        _, out_positions = get_top_sharpest_patches_fast(sharpness_map, hr_image_np, patch_size, top_n)

        hr_patches = [hr_image_np[y:y+patch_size, x:x+patch_size] for y, x in out_positions]
        lr_patches = crop_lr_patches(lr_image_np, out_positions, patch_size, scale_factor)

        # Save patches
        for i, (hr_patch, lr_patch) in enumerate(zip(hr_patches, lr_patches)):
            hr_save_path = os.path.join(hr_save_dir, f"{hr_image_name[:-4]}_patch_{i}.png")
            lr_save_path = os.path.join(lr_save_dir, f"{hr_image_name[:-4]}_patch_{i}.png")

            cv2.imwrite(hr_save_path, cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(lr_save_path, cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))

def get_ccorr_normed(
    image: np.ndarray, target: np.ndarray, interpolation=cv2.INTER_CUBIC
) -> float:
    """Get the normalized cross correlation between two images.

    Args:
        image (np.ndarray): Source image.
        target (np.ndarray): Target image, if target image is not the
            same size as the source image, resize it.
        interpolation (int, optional): Resize interpolation method.
            Defaults to cv.INTER_CUBIC.

    Returns:
        float: _description_
    """
    return cv2.matchTemplate(
        image,
        cv2.resize(
            target, (image.shape[1], image.shape[0]), interpolation=interpolation
        ),
        cv2.TM_CCORR_NORMED,
    )[0][0]


def compute_ranked_correlation(hr_dir, lr_dir):
    hr_images = os.listdir(hr_dir)
    correlation_scores = []

    for hr_img_name in hr_images:
        hr_img_path = os.path.join(hr_dir, hr_img_name)
        lr_img_path = os.path.join(lr_dir, hr_img_name)

        hr_img_np = cv2.imread(hr_img_path, cv2.IMREAD_COLOR)
        lr_img_np = cv2.imread(lr_img_path, cv2.IMREAD_COLOR)

        score = get_ccorr_normed(hr_img_np, lr_img_np)
        correlation_scores.append((score, hr_img_name))

    # Sorting by scores in descending order
    correlation_scores.sort(reverse=True, key=lambda x: x[0])

    return [img_name for _, img_name in correlation_scores]

def retain_top_images(hr_dir, lr_dir, top_n=100):
    ranked_images = compute_ranked_correlation(hr_dir, lr_dir)
    for idx, img_name in enumerate(ranked_images):
        if idx >= top_n:
            os.remove(os.path.join(hr_dir, img_name))
            os.remove(os.path.join(lr_dir, img_name))

# Paths
hr_path = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/HR/'
lr_path = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/LR/x4/'
hr_save_dir = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/HR_meta/'
lr_save_dir = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/LR_meta/x4/'

process_and_save_images(hr_path, lr_path, hr_save_dir, lr_save_dir)

HR_PATH = hr_save_dir
LR_PATH = lr_save_dir

retain_top_images(HR_PATH, LR_PATH)
