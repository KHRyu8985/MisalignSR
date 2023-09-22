import gradio
import gradio as gr
from gradio import Interface
from gradio.components import Image as grImage, Dropdown, Slider, Checkbox, Radio

from PIL import Image, ImageDraw
from PIL import ImageOps


def resize_image(image, base_width):
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    return image.resize((base_width, h_size), Image.BICUBIC)


def checkerboard_image_gr(img1, img2, square_size=70, show_grid=True):
    # Resize the images
    img1 = resize_image(img1, 700)
    img2 = resize_image(img2, 700)

    # Initialize a blank image for the checkerboard pattern
    result = Image.new("RGB", img1.size)

    # Iterate over the image in steps of square size
    for i in range(0, img1.width, square_size):
        for j in range(0, img1.height, square_size):
            # Determine which image to use based on the position
            source_img = (
                img1 if (i // square_size) % 2 == (j // square_size) % 2 else img2
            )
            patch = source_img.crop((i, j, i + square_size, j + square_size))
            result.paste(patch, (i, j))

    # If grid option is selected, overlay a grid
    if show_grid:
        draw = ImageDraw.Draw(result)
        for i in range(0, result.width, square_size):
            draw.line([(i, 0), (i, result.height)], fill="white", width=1)
        for i in range(0, result.height, square_size):
            draw.line([(0, i), (result.width, i)], fill="white", width=1)

    return result


# Update the gradio interface function to use the checkerboard function
def gr_interface(img1, img2, square_size=70, show_grid=True):
    return checkerboard_image_gr(img1, img2, square_size, show_grid)


# Update the gradio Interface to use the updated function
Interface(
    fn=gr_interface,
    inputs=[
        grImage(type="pil", label="LR image"),
        grImage(type="pil", label="HR image"),
        Slider(10, 700, label="Square Size", step=10, default=100),
        Checkbox(label="Show Grid")
    ],
    outputs=grImage(type="pil"),
    css=".output_image img { width: 100%; height: auto; }",
    live=True
).launch()
