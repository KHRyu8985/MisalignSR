import cv2
import os
import argparse

# Run like:
# python scripts/data/zoom/prepare_equalize.py /home/kanghyun/MisalignSR/datasets/Zoom-to-Learn /home/kanghyun/MisalignSR/datasets/ZOOM
def process_images(folder_name, base_folder, output_folder):
    normalized_path = os.path.join(base_folder, folder_name, 'normalized')
    images = sorted(os.listdir(normalized_path))

    if len(images) < 6:
        return

    # Assuming the first image is HR
    hr_image_path = os.path.join(normalized_path, images[0])
    image_hr = cv2.imread(hr_image_path)
    hr_h, hr_w, _ = image_hr.shape

    # Resize HR to be multiple of 480
    new_hr_w = (hr_w // 480) * 480
    new_hr_h = (hr_h // 480) * 480
    image_hr_resized = cv2.resize(image_hr, (new_hr_w, new_hr_h), interpolation=cv2.INTER_AREA)

    # Save to HR directory
    hr_save_path = os.path.join(output_folder, 'HR', f'{folder_name}.png')
    if not os.path.exists(os.path.join(output_folder, 'HR')):
        os.makedirs(os.path.join(output_folder, 'HR'))
    cv2.imwrite(hr_save_path, image_hr_resized)

    # Resizing logic for other images
    scales = [2, 3, 4, 5]
    sub_dirs = ['x2', 'x3', 'x4', 'x5']

    for idx, (sub_dir, scale) in enumerate(zip(sub_dirs, scales)):
        if idx >= len(images):  # To avoid index out of range if there are fewer images
            break

        new_width = int(new_hr_w / scale)
        new_height = int(new_hr_h / scale)

        lr_image_path = os.path.join(normalized_path, images[idx])
        image_lr = cv2.imread(lr_image_path)
        image_lr_resized = cv2.resize(image_lr, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create directory if it doesn't exist
        lr_save_dir = os.path.join(output_folder, 'LR', sub_dir)
        if not os.path.exists(lr_save_dir):
            os.makedirs(lr_save_dir)

        # Save the resized image
        lr_save_path = os.path.join(lr_save_dir, f'{folder_name}.png')
        cv2.imwrite(lr_save_path, image_lr_resized)

def main():
    parser = argparse.ArgumentParser(description='Process images and save to designated ZOOM directory.')
    parser.add_argument('base_path', type=str, help='Base path to Zoom-to-Learn directory.')
    parser.add_argument('zoom_path', type=str, help='Path where the ZOOM folder will be created.')
    args = parser.parse_args()

    train_path = os.path.join(args.base_path, 'train', 'train')
    test_path = os.path.join(args.base_path, 'test', 'test')

    # Processing train images
    train_output_path = os.path.join(args.zoom_path, 'train')
    train_folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    for folder in train_folders:
        process_images(folder, train_path, train_output_path)

    # Processing test images
    test_output_path = os.path.join(args.zoom_path, 'test')
    test_folders = [f for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))]
    for folder in test_folders:
        process_images(folder, test_path, test_output_path)

if __name__ == "__main__":
    main()
