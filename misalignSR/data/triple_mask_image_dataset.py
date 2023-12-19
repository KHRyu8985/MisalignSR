from logging import raiseExceptions
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from misalignSR.data.data_util import (multiple_paths_from_meta_info_file,
                                    multiple_paths_from_folder)
from misalignSR.data.transforms import augment, random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TripleMaskImageDataset(data.Dataset):
    """Triple image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT, Reference image triplet.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.ref_folder, self.mask_folder = opt['dataroot_ref'], opt['dataroot_mask']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            # self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.gt_folder]
            # self.io_backend_opt['client_keys'] = ['lq', 'gt']
            # self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            raise NotImplementedError(f'backend type lmdb not implemented yet.')
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = multiple_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.ref_folder, self.mask_folder],
                ['lq', 'gt', 'ref', 'mask'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = multiple_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.ref_folder, self.mask_folder],
                ['lq', 'gt', 'ref', 'mask'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = imfrombytes(img_bytes, float32=True)

        mask_path = self.paths[index]['mask_path']
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_ref, img_mask = random_crop([img_gt, img_lq,
                                                             img_ref, img_mask], gt_size)

            # flip, rotation
            img_gt, img_lq, img_ref, img_mask = augment([img_gt, img_lq,
                                                         img_ref, img_mask], self.opt['use_flip'], self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_ref, img_mask = img2tensor([img_gt, img_lq, img_ref, img_mask],
                                                       bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {'lq': img_lq,
                'gt': img_gt,
                'ref': img_ref,
                'mask': img_mask,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'ref_path': ref_path,
                'mask_path': mask_path}

    def __len__(self):
        return len(self.paths)
