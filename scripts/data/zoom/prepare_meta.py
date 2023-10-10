import os
import shutil
import random
import pyrootutils
from pathlib import Path


def main():

    cwd = Path().resolve()
    rootdir = pyrootutils.setup_root(
        search_from=cwd,
        indicator='.project-root',
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )  # root: root folder path of the directory
    rootdir = os.path.join(rootdir, 'datasets/ZOOM')

    print('Path for the root folder is:')
    print(rootdir)

    hr_train = os.path.join(rootdir, 'ZOOM_train_HR_sub')
    meta_hr = os.path.join(rootdir, 'ZOOM_meta_HR_sub')

    lr_train_roots = ['ZOOM_train_LR_bicubic/X2_sub', 'ZOOM_train_LR_bicubic/X3_sub', 'ZOOM_train_LR_bicubic/X4_sub']
    meta_lr_roots = ['ZOOM_meta_LR_bicubic/X2_sub', 'ZOOM_meta_LR_bicubic/X3_sub', 'ZOOM_meta_LR_bicubic/X4_sub']

    if not os.path.exists(meta_hr):
        os.makedirs(meta_hr)

    for meta_lr in meta_lr_roots:
        if not os.path.exists(os.path.join(rootdir, meta_lr)):
            os.makedirs(os.path.join(rootdir, meta_lr))

    hr_files = [f for f in os.listdir(hr_train) if f.endswith('.png')]
    # hr_files = [f for f in os.listdir(hr_train) if f.endswith('.JPG')]
    random.shuffle(hr_files)
    subset_hr_files = hr_files[:100]  # Change 100 to the number of files you want

    # Copy HR files
    for file in subset_hr_files:
        shutil.copy(os.path.join(hr_train, file), os.path.join(meta_hr, file))

    # Copy corresponding LR files
    for file in subset_hr_files:
        for lr_train, meta_lr in zip(lr_train_roots, meta_lr_roots):
            src_path = os.path.join(rootdir, lr_train, file)
            dest_path = os.path.join(rootdir, meta_lr, file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)


if __name__ == '__main__':
    main()
