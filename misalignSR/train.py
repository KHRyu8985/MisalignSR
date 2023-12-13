import os
import os.path as osp
import click

import misalignSR.archs
import misalignSR.data
import misalignSR.losses
import misalignSR.models
from basicsr.train import train_pipeline
import sys

def list_option_files():
    options_path = osp.join(osp.dirname(osp.dirname(__file__)), 'options', 'train')
    yaml_files = []
    for root, dirs, files in os.walk(options_path):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                # Construct the relative path to display to the user
                relative_path = osp.relpath(osp.join(root, file), options_path)
                yaml_files.append(relative_path)
    return yaml_files

@click.command()
def interactive_train():
    click.echo("Select a training option:")
    option_files = list_option_files()
    option_files.sort()
    for i, file in enumerate(option_files):
        if i < 9:
            click.echo(f"0{i + 1}: {file}")
        else:
            click.echo(f"{i + 1}: {file}")

    choice = click.prompt("Please enter the number of your choice", type=int)
    selected_option = option_files[choice - 1]
    click.echo(f"You selected: {selected_option}")

    if click.confirm('Do you want to continue with this option?'):
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
        option_file = osp.join('options', 'train', selected_option)

        # Simulate command-line arguments
        sys.argv = ['train.py', '-opt', option_file]
        train_pipeline(root_path)
    else:
        click.echo("Training cancelled.")

if __name__ == '__main__':
    interactive_train()
