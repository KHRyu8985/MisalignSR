import os
import argparse
from shutil import copy2

# Function to copy and rename files


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


def main(original_dataset_path, new_dataset_path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and process dataset for MisalignSR.')
    parser.add_argument('--original_dataset_path', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--new_dataset_path', type=str, required=True, help='Path to the new dataset')

    args = parser.parse_args()

    new_path = main(args.original_dataset_path, args.new_dataset_path)
    print(f"Dataset processed and stored at: {new_path}")

