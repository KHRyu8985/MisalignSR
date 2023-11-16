import os
import numpy as np
from PIL import Image
import argparse
from defocus.cpbd import compute
from multiprocessing import Pool
from tqdm import tqdm

def cpbd_sharpness_score(im):
    image_np = np.asarray(im)
    image_grayscale = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])
    score = compute(image_grayscale)
    return score

def process_image(file):
    try:
        im = Image.open(file)
        score = cpbd_sharpness_score(im)
        return (file, score)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return (file, 0)

def process_images(source_folder, destination_folder, num_images=100, worst=False):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.png')]

    with Pool() as pool:
        scores = list(tqdm(pool.imap(process_image, files), total=len(files), desc='Computing CPBD scores'))

    # Sort based on worst flag
    top_files = sorted(scores, key=lambda x: x[1], reverse=not worst)[:num_images]

    for file, _ in top_files:
        filename = os.path.basename(file)
        im = Image.open(file)
        im.save(os.path.join(destination_folder, filename))

def main():
    parser = argparse.ArgumentParser(description='Process images for CPBD sharpness score.')
    parser.add_argument('--source', type=str, required=True, help='Path to the source folder containing images')
    parser.add_argument('--dest', type=str, required=True, help='Path to the destination folder to save top images')
    parser.add_argument('--num', type=int, default=100, help='Number of top/bottom sharp images to save')
    parser.add_argument('--worst', action='store_true', help='Set to save the least sharp images')

    args = parser.parse_args()

    process_images(args.source, args.dest, args.num, args.worst)

if __name__ == "__main__":
    main()
