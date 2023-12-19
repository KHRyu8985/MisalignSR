import os
import gradio as gr
from PIL import Image

# Define the directories for HR and LR images
hr_image_dir = '/home/kanghyun/MisalignSR/results/RRDB_x3_LRE_ZOOM/visualization/ZOOM'
lr_image_dir = '/home/kanghyun/MisalignSR/results/RRDB_x3_ZOOM/visualization/ZOOM'

def find_matching_image(filename, search_dir):
    """
    Find the matching image in the specified directory based on the common filename pattern.
    """
    common_name_part = filename.split('_')[:2]  # Extracting the common part of the filename
    for file in os.listdir(search_dir):
        if all(part in file for part in common_name_part):
            return file
    return None

def display_images(image_name):
    """
    Display HR and corresponding LR images side by side.
    """
    hr_image_path = os.path.join(hr_image_dir, image_name)
    lr_image_name = find_matching_image(image_name, lr_image_dir)
    if lr_image_name:
        lr_image_path = os.path.join(lr_image_dir, lr_image_name)
        hr_image = Image.open(hr_image_path)
        lr_image = Image.open(lr_image_path)
        return hr_image, lr_image
    else:
        return "No matching LR image found."

# List of HR image filenames for dropdown
hr_images = [file for file in os.listdir(hr_image_dir) if file.endswith('.png') or file.endswith('.jpg')]

# Gradio interface
iface = gr.Interface(
    fn=display_images,
    inputs=gr.Dropdown(choices=hr_images),
    outputs=[gr.Image(type="pil"), gr.Image(type="pil")],
    title="HR and LR Image Viewer"
)

iface.launch()
