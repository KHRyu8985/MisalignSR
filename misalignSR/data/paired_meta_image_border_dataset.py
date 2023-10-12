from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import (
    augment,
    paired_random_crop,
    paired_random_crop_with_border,
)
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedBorderMetaImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedBorderMetaImageDataset, self).__init__()
        self.border = 5  # added border
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        self.meta_gt_folder, self.meta_lq_folder = (
            opt["dataroot_meta_gt"],
            opt["dataroot_meta_lq"],
        )  # added meta folders

        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl
        )

        self.meta_paths = paired_paths_from_folder(
            [self.meta_lq_folder, self.meta_gt_folder],
            ["meta_lq", "meta_gt"],
            self.filename_tmpl,
        )  # added meta paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]["lq_path"]
        img_bytes = self.file_client.get(lq_path, "lq")
        img_lq = imfrombytes(img_bytes, float32=True)

        meta_index = index % len(self.meta_paths)
        meta_gt_path = self.meta_paths[meta_index]["meta_gt_path"]  # added meta path
        img_bytes = self.file_client.get(meta_gt_path, "meta_gt")
        meta_img_gt = imfrombytes(img_bytes, float32=True)

        meta_lq_path = self.meta_paths[meta_index]["meta_lq_path"]  # added meta path
        img_bytes = self.file_client.get(meta_lq_path, "meta_lq")
        meta_img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = paired_random_crop_with_border(
                img_gt, img_lq, gt_size, scale, border=self.border, gt_path=gt_path,
            )
            meta_img_gt, meta_img_lq = paired_random_crop(
                meta_img_gt, meta_img_lq, gt_size, scale, meta_gt_path
            )

            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )

            meta_img_gt, meta_img_lq = augment(
                [meta_img_gt, meta_img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )

        # color space transform
        if "color" in self.opt and self.opt["color"] == "y":
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
            meta_img_gt = bgr2ycbcr(meta_img_gt, y_only=True)[..., None]
            meta_img_lq = bgr2ycbcr(meta_img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt["phase"] != "train":
            img_gt = img_gt[
                0 : img_lq.shape[0] * scale + 2 * self.border,
                0 : img_lq.shape[1] * scale + 2 * self.border,
                :,
            ]
            meta_img_gt = meta_img_gt[
                0 : meta_img_lq.shape[0] * scale, 0 : meta_img_lq.shape[1] * scale, :
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        meta_img_gt, meta_img_lq = img2tensor(
            [meta_img_gt, meta_img_lq], bgr2rgb=True, float32=True
        )

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

            normalize(meta_img_lq, self.mean, self.std, inplace=True)
            normalize(meta_img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "meta_lq": meta_img_lq,
            "meta_gt": meta_img_gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)
