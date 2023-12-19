#!/bin/bash

# Navigate to the directory containing the files.
cd /home/kanghyun/MisalignSR/results/Alignformer_test/visualization/ZOOM

# Loop through all .png files and rename them by stripping out the pattern '_*_*'.
for file in *.png; do
  # Use parameter expansion to construct the new filename.
  base=${file%%_*}  # This strips everything after the first underscore.
  suffix=${file#*_s}  # This strips everything before '_s' including it.
  suffix=${suffix%%_*}.png  # This strips everything after the second underscore in the original suffix.
  newname="${base}_s${suffix}"  # This constructs the new filename.

  mv "$file" "$newname"  # Renames the file.
done
