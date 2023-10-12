import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from misalignSR.losses.basic_loss import GradientLoss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

import higher
import numpy as np

@MODEL_REGISTRY.register()
class LRESRModel(SRModel):
    """Misaligned Meta learning SR model for single image super-resolution."""

    """Modify from original SRModel in basicsr/models/SRModel.py"""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.weight_vis = self.opt["train"].get("weight_vis", True)

        self.meta_loss_type = self.opt["train"].get("meta_loss", None) # if True then use GDL, else use L1
        self.start_meta = self.opt["train"].get("start_meta", None)

        if self.meta_loss_type is None:
            self.meta_loss_type = 'GDL' # Default to GDL
        if self.start_meta is None:
            self.start_meta = -1

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        if "meta_lq" in data:
            self.meta_lq = data["meta_lq"].to(self.device)
            self.meta_gt = data["meta_gt"].to(self.device)

    def determine_meta_weight(self):

        if self.meta_loss_type == 'GDL':
            loss_func = GradientLoss(reduction="none").to(self.device)
        else:
            loss_func = torch.nn.L1Loss(reduction="none").to(self.device)

        with higher.innerloop_ctx(
            self.net_g, self.optimizer_g, copy_initial_weights=True
        ) as (fnet, diffopt):
            # 1. Update meta model on training data
            output = fnet(self.lq)
            meta_train_loss = torch.mean(loss_func(output, self.gt), dim=1, keepdim=True)

            # set pixel-reweight method
            eps = torch.zeros(
                meta_train_loss.size(), requires_grad=True, device=self.device
            )

            meta_train_loss = torch.sum(eps * meta_train_loss)
            diffopt.step(meta_train_loss)

            # 2. Compute grads of eps on meta validation data
            meta_val_loss = loss_func(fnet(self.meta_lq), self.meta_gt)
            meta_val_loss = torch.mean(meta_val_loss)

            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

        # Compute weights for current training batch
        meta_weights = torch.clamp(-eps_grads, min=0)
        l1_norm = torch.mean(meta_weights)
        if l1_norm != 0:
            meta_weights = meta_weights / l1_norm
        else:
            meta_weights = meta_weights
        return meta_weights

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if current_iter > self.start_meta:
            weights = self.determine_meta_weight()
        else:
            weights = 1.0
        self.output = self.net_g(self.lq)

        if self.weight_vis and current_iter % 5000 == 1 and current_iter > self.start_meta:
            logger = get_root_logger()
            logger.info(f"Visualize weights at iteration {current_iter}")
            logger.info(f"weight is Mean: {weights.mean()}, STD: {weights.std()}")
            logger.info(f"Loss type is {self.meta_loss_type}")

            w = tensor2img(weights.detach().cpu())
            o = tensor2img(self.output.detach().cpu())
            g = tensor2img(self.gt.detach().cpu())
            i = np.concatenate((o, w, g), axis=1)

            save_img_path = osp.join(
                self.opt["path"]["visualization"], "weights", f"{current_iter}.png"
            )

            imwrite(i, save_img_path)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(weights * self.output, weights * self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
