import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from misalignSR.losses.basic_loss import GWLoss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

import numpy as np
import torchopt


@MODEL_REGISTRY.register()
class LRESRModel(SRModel):
    """Misaligned Meta learning SR model for single image super-resolution."""

    """Modify from original SRModel in basicsr/models/SRModel.py"""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.weight_vis = self.opt["train"].get("weight_vis", True)
        self.lre_batch_only = self.opt["train"].get("lre_batch_only", False)

        self.meta_loss_type = self.opt["train"].get("meta_loss", None)  # if True then use GDL, else use L1
        self.start_meta = self.opt["train"].get("start_meta", None)

        if self.meta_loss_type is None:
            self.meta_loss_type = 'GW'  # Default to GDL
        if self.start_meta is None:
            self.start_meta = -1

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        lr = self.opt['train']['optim_g'].get("lr", 1e-4)
        self.model_optimizer = torchopt.MetaSGD(self.net_g, lr=lr)  # use it for LRE

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

        self.test_all = self.opt['val'].get('test_all', False)
        self.save_csv = self.opt['val'].get('save_csv', False)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        if "meta_lq" in data:
            self.meta_lq = data["meta_lq"].to(self.device)
            self.meta_gt = data["meta_gt"].to(self.device)

    def determine_meta_weight(self):
        # train pair: self.lq, self.gt
        # meta pair: self.meta_lq, self.meta_gt
        if self.meta_loss_type == 'GW':
            loss_func = GWLoss(reduction="none").to(self.device)
        else:
            loss_func = torch.nn.L1Loss(reduction="none").to(self.device)

        # First we need to reset the meta_weights to 0 (per-example)
        meta_weights = torch.zeros((self.lq.shape[0], 1, 1, 1), requires_grad=True, device=self.device)

        # Now we save theta_{t-1} state of network and optimizer
        net_state_dict = torchopt.extract_state_dict(self.net_g)
        optim_state_dict = torchopt.extract_state_dict(self.model_optimizer)

        # Now we perform the inner loop
        for _ in range(1):
            train_sr = self.net_g(self.lq)
            inner_loss = torch.sum(torch.mean(loss_func(train_sr, self.gt), dim=(1, 2, 3), keepdim=True) * meta_weights)
            self.model_optimizer.step(inner_loss)

        # Now we perform the outer loop
        meta_sr = self.net_g(self.meta_lq)
        outer_loss = torch.mean(loss_func(meta_sr, self.meta_gt))
        meta_weights = -torch.autograd.grad(outer_loss, meta_weights)[0]
        # Normalize the weights
        meta_weights = torch.nn.ReLU()(meta_weights.detach())
        weights_sum = torch.sum(meta_weights)
        weights_sum = weights_sum + 1 if weights_sum == 0 else weights_sum
        meta_weights = meta_weights / weights_sum

        # Now we restore the theta_{t-1} state of network and optimizer
        torchopt.recover_state_dict(self.net_g, net_state_dict)
        torchopt.recover_state_dict(self.model_optimizer, optim_state_dict)

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
