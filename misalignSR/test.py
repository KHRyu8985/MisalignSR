import os
import os.path as osp
import click

import misalignSR.archs
import misalignSR.data
import misalignSR.losses
import misalignSR.models
from basicsr.test import test_pipeline
import sys

def list_option_files():
    options_path = osp.join(osp.dirname(osp.dirname(__file__)), 'options', 'test')
    yaml_files = []
    for root, dirs, files in os.walk(options_path):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                # Construct the relative path to display to the user
                relative_path = osp.relpath(osp.join(root, file), options_path)
                yaml_files.append(relative_path)
    return yaml_files

@click.command()
def interactive_test():
    click.echo("Select a testing option:")
    option_files = list_option_files()
    for i, file in enumerate(option_files):
        click.echo(f"{i + 1}: {file}")

    choice = click.prompt("Please enter the number of your choice", type=int)
    selected_option = option_files[choice - 1]
    click.echo(f"You selected: {selected_option}")

    if click.confirm('Do you want to continue with this option?'):
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
        option_file = osp.join('options', 'test', selected_option)

        # Simulate command-line arguments
        sys.argv = ['test.py', '-opt', option_file]
        test_pipeline(root_path)
    else:
        click.echo("Training cancelled.")

if __name__ == '__main__':
    interactive_test()
