from utils import find_kernel, filtering
import matplotlib.image as img
import matplotlib.pyplot as plt
from basicsr.utils import img2tensor, tensor2img

from KernelGAN.kernelGAN import KernelGAN
from KernelGAN.data import DataGenerator
from KernelGAN.learner import Learner
from KernelGAN.configs import Config
from KernelGAN.util import save_final_kernel, run_zssr, post_process_k
from KernelGAN.ZSSRforKernelGAN.ZSSR import ZSSR

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# running the code example
# python notebooks/run_kernelgan_zssr.py --gt_path '/home/kanghyun/MisalignSR/datasets/DIV2K/DIV2K_train_HR_sub/0001_s019.png' --lr_path '/home/kanghyun/MisalignSR/datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub/0001_s019.png' --output_dir '/home/kanghyun/MisalignSR/notebooks/figures'


def plot_and_save(image, title, filename):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def kernel_estimation_model_based(gt_tensor, lr_tensor):
    kernel = find_kernel(gt_tensor, lr_tensor, scale=2, k=20, max_patches=-1)
    return kernel


def kernel_estimation_gan(conf):
    kernel_gan_net = KernelGAN(conf)
    datagen = DataGenerator(conf, kernel_gan_net)
    learner = Learner()
    for iteration in tqdm(range(100), ncols=60):
        [g_in, d_in] = datagen.__getitem__(iteration)
        kernel_gan_net.train(g_in, d_in)
        learner.update(iteration, kernel_gan_net)
    kernels = kernel_gan_net.curr_k
    kernel_final = post_process_k(kernels, 40)
    return kernel_final


def apply_ZSSR(input_image_path, kernel_final):
    sr = ZSSR(
        input_image_path,
        scale_factor=2,
        kernels=[kernel_final],
        is_real_img=False,
        noise_scale=1,
    ).run()
    return sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--lr_path", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    gt = img.imread(args.gt_path)
    lr = img.imread(args.lr_path)

    import os

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # make output directory if it doesn't exist

    plot_and_save(lr, "LR", f"{args.output_dir}/LR.png")
    plot_and_save(gt, "GT", f"{args.output_dir}/GT.png")

    lr_tensor = img2tensor(lr, bgr2rgb=False)
    gt_tensor = img2tensor(gt, bgr2rgb=False)

    kernel_model_based = kernel_estimation_model_based(gt_tensor, lr_tensor)
    plot_and_save(
        kernel_model_based,
        "Kernel Model Based",
        f"{args.output_dir}/Kernel_Model_Based.png",
    )

    class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    # Sample dictionary
    conf_dict = {
        "img_name": "image1",
        "input_image_path": args.lr_path,
        "input_crop_size": 64,
        "scale_factor": 0.5,
        "X4": False,
        "G_chan": 64,
        "D_chan": 64,
        "G_kernel_size": 13,
        "D_n_layers": 7,
        "D_kernel_size": 7,
        "max_iters": 3000,
        "g_lr": 2e-4,
        "d_lr": 2e-4,
        "beta1": 0.5,
        "gpu_id": 0,
        "n_filtering": 40,
        "do_ZSSR": False,
        "noise_scale": 1.0,
        "real_image": False,
        "G_structure": [7, 5, 3, 1, 1, 1],
    }

    # Convert the dictionary to a class instance
    conf = Config(conf_dict)

    kernel_gan = kernel_estimation_gan(conf)
    plot_and_save(kernel_gan, "Kernel GAN", f"{args.output_dir}/Kernel_GAN.png")

    sr = apply_ZSSR(args.lr_path, kernel_gan)
    plot_and_save(sr, "KERNEL+ZSSR", f"{args.output_dir}/KG_ZSSR.png")


if __name__ == "__main__":
    main()
