import gradio as gr
import os
from PIL import Image, ImageDraw

# Define the directories for HR and LR images
hr_image_dir = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/HR_sub'
lr_image_dir = '/home/kanghyun/MisalignSR/datasets/ZOOM/train/LR/4_sub'

# Get the sorted list of images
hr_images = sorted(os.listdir(hr_image_dir))

# Function to create a curtain effect with a diagonal line
def curtain_effect(img1, img2, curtain_position):
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    result = Image.new('RGB', img1.size)
    draw = ImageDraw.Draw(result)

    # Calculate the x-intercept for the 45-degree line based on curtain_position
    b = curtain_position * (img1.width + img1.height)

    # Create the curtain effect
    for x in range(img1.width):
        for y in range(img1.height):
            if y < (-x + b):  # Pixel is above the 45-degree line
                pixel = img1.getpixel((x, y))
            else:
                pixel = img2.getpixel((x, y))
            result.putpixel((x, y), pixel)

    # Draw the line at the curtain
    if b <= img1.width:  # Line is in the top triangle
        x_pos = int(b)
        y_pos = 0
    else:  # Line is in the bottom triangle
        x_pos = img1.width
        y_pos = int(b - img1.width)
    draw.line([(x_pos, y_pos), (0, b)], fill=(255, 0, 0), width=2)

    return result

# Function to handle the Gradio interface logic
def update_interface(index, curtain_position):

    # Get image paths
    hr_image_path, lr_image_path = get_image_paths(index)
    hr_image = Image.open(hr_image_path)
    lr_image = Image.open(lr_image_path)

    # Generate curtain effect image with the specified curtain position
    curtain_img = curtain_effect(hr_image, lr_image, curtain_position)

    # Return the original and curtain effect images
    return hr_image, lr_image, curtain_img

# Function to get HR and LR image paths based on index
def get_image_paths(index):
    return os.path.join(hr_image_dir, hr_images[index]), os.path.join(lr_image_dir, hr_images[index])

# Function to save the flag
def flag_image(index):
    filename = hr_images[index]
    flagged = False  # A flag to check if the image has been flagged already

    # Check if the file exists and read its contents
    if os.path.isfile('flagged_images.txt'):
        with open('flagged_images.txt', 'r') as file:
            if filename + '\n' in file.readlines():
                flagged = True

    # If the image has not been flagged, write the filename to the file
    if not flagged:
        with open('flagged_images.txt', 'a') as file:
            file.write(filename + '\n')
        print(f"Image {filename} flagged.")
        return "Image flagged."
    else:
        print(f"Image already flagged.")
        return "Image already flagged."

# Create the Gradio app
with gr.Blocks() as app:
    with gr.Row():
        index_slider = gr.Slider(0, len(hr_images) - 1, step=1, label="Image Index")
        curtain_position_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label='Curtain Position')
        flag_button = gr.Button("Flag Image")
    with gr.Row():
        hr_image_component = gr.Image(label="HR Image")
        lr_image_component = gr.Image(label="LR Image")
        blended_image_component = gr.Image(label="Blended Image")

    index_slider.change(fn=update_interface, inputs=[index_slider, curtain_position_slider], outputs=[hr_image_component, lr_image_component, blended_image_component])
    curtain_position_slider.change(fn=update_interface, inputs=[index_slider, curtain_position_slider], outputs=[hr_image_component, lr_image_component, blended_image_component])
    flag_button.click(fn=flag_image, inputs=[index_slider])

app.launch()
