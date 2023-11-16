#!/bin/bash

# Define the source and destination directories for HR and LR images
SOURCE_HR_DIR="datasets/ZOOM/train/HR_sub"
SOURCE_LR_DIR="datasets/ZOOM/train/LR/4_sub"
DEST_HR_META_DIR="datasets/ZOOM/train/HR_4_meta"
DEST_LR_META_DIR="datasets/ZOOM/train/LR_4_meta"

# Create destination directories if they don't exist
mkdir -p "$DEST_HR_META_DIR"
mkdir -p "$DEST_LR_META_DIR"

# Path to the text file containing filenames
FILENAME_LIST="flagged_images.txt"

# Read filenames from the text file and copy HR images
while IFS= read -r FILENAME; do
    cp "$SOURCE_HR_DIR/$FILENAME" "$DEST_HR_META_DIR/"
    echo "$FILENAME"
done < "$FILENAME_LIST"

echo "HR images copied"
# Read filenames from the text file and copy LR images
while IFS= read -r FILENAME; do
    cp "$SOURCE_LR_DIR/$FILENAME" "$DEST_LR_META_DIR/"
    echo "$FILENAME"

echo "LR images copied"

done < "$FILENAME_LIST"
