import gradio
import gradio as gr
from gradio import Interface
from gradio.components import Image as grImage, Dropdown, Slider, Checkbox, Radio

from PIL import Image, ImageDraw

def resize_image(image, base_width):
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    return image.resize((base_width, h_size), Image.BICUBIC)

def image_curtain_gr(img1, img2, direction='horizontal', curtain_position=0.5, show_grid=True, grid_spacing=70):
    img1 = resize_image(img1, 700)
    img2 = resize_image(img2, 700)

    def add_grid_to_image(image, spacing):
        if not show_grid:
            return image

        img_with_grid = image.copy()
        draw = ImageDraw.Draw(img_with_grid)
        dot_length = 1
        space_length = 5

        # Function to draw dotted lines
        def draw_dotted_line(start, end):
            total_length = max(abs(start[0] - end[0]), abs(start[1] - end[1]))
            for i in range(0, total_length, dot_length + space_length):
                if start[0] == end[0]:  # vertical line
                    line_start = (start[0], start[1] + i)
                    line_end = (end[0], start[1] + i + dot_length)
                else:  # horizontal line
                    line_start = (start[0] + i, start[1])
                    line_end = (start[0] + i + dot_length, end[1])
                draw.line([line_start, line_end], fill=(255, 255, 255), width=1)

        for x in range(0, image.width, spacing):
            draw_dotted_line((x, 0), (x, image.height))
        for y in range(0, image.height, spacing):
            draw_dotted_line((0, y), (image.width, y))
        return img_with_grid

    img1 = add_grid_to_image(img1, grid_spacing)
    img2 = add_grid_to_image(img2, grid_spacing)

    result = Image.new('RGB', img1.size)

    draw = ImageDraw.Draw(result)

    if direction == 'horizontal':
        split_at = int(curtain_position * img1.width)
        result.paste(img1, (0, 0))
        result.paste(img2.crop((split_at, 0, img2.width, img2.height)), (split_at, 0))
        draw.line([(split_at, 0), (split_at, img1.height)], fill=(255, 0, 0), width=2)
    elif direction == 'vertical':
        split_at = int(curtain_position * img1.height)
        result.paste(img1, (0, 0))
        result.paste(img2.crop((0, split_at, img2.width, img2.height)), (0, split_at))
        draw.line([(0, split_at), (img1.width, split_at)], fill=(255, 0, 0), width=2)
    else:  # diagonal
        # Calculate the x-intercept for the 45-degree line based on curtain_position
        b = curtain_position * (img1.width + img1.height)

        for x in range(img1.width):
            for y in range(img1.height):
                if y < (-x + b):  # Check if the pixel is above the 45-degree line
                    pixel = img1.getpixel((x, y))
                else:
                    pixel = img2.getpixel((x, y))
                result.putpixel((x, y), pixel)

        # Calculate the line's end position
        if b <= img1.width:  # Line is in the top triangle
            x_pos = int(b)
            y_pos = 0
        else:  # Line is in the bottom triangle
            x_pos = img1.width
            y_pos = int(b - img1.width)

        draw.line([(x_pos, y_pos), (0, b)], fill=(255, 0, 0), width=2)

    return result


def gr_interface(img1, img2, direction=None, curtain_position=None, show_grid=None, grid_spacing=None):
    if img1 is None or img2 is None:
        blank_image = Image.new('RGB', (700, 700), color='white')
        return blank_image

    # Set default values
    if direction is None:
        direction = 'diagonal'
    if curtain_position is None:
        curtain_position = 0.5
    if show_grid is None:
        show_grid = True
    if grid_spacing is None:
        grid_spacing = 70

    result = image_curtain_gr(img1, img2, direction, curtain_position, show_grid, grid_spacing)
    return result

# UI Components
direction_dropdown = Dropdown(choices=['horizontal', 'vertical', 'diagonal'], default='diagonal', label='Direction')
curtain_position_slider = Slider(minimum=0, maximum=1, step=0.01, default=0.5, label='Curtain Position')
show_grid_checkbox = Checkbox(default=None, label='Show Grid')
grid_spacing_slider = Slider(minimum=10, maximum=200, step=5, default=70, label='Grid Spacing')

# Launch the Gradio UI
Interface(
    fn=gr_interface,
    inputs=[
        grImage(type='pil', label='LR image'),
        grImage(type='pil', label='HR image'),
        direction_dropdown,
        curtain_position_slider,
        show_grid_checkbox,
        grid_spacing_slider
    ],
    outputs=grImage(type='pil'),
    css='.output_image img { width: 100%; height: auto; }',
    live=True
).launch()
