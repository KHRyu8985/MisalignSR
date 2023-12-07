import torch
import torch.nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
import higher
import numpy as np
from misalignSR.losses.basic_loss import GradientLoss


@MODEL_REGISTRY.register()
class MWNSRModel(SRModel):
    """Misaligned Meta learning SR model for single image super-resolution."""

    """Modify from original SRModel in basicsr/models/SRModel.py"""

    def init_training_settings(self):
        train_opt = self.opt["train"]
        self.weight_vis = train_opt.get("weight_vis", True)

        if self.weight_vis:
            logger = get_root_logger()
            logger.info("Visualize weights")
        else:
            logger = get_root_logger()

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_v
        self.net_v = build_network(self.opt["network_mwn"])
        self.net_v = self.model_to_device(self.net_v)
        self.print_network(self.net_v)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_v", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_v", "params")
            self.load_network(
                self.net_v,
                load_path,
                self.opt["path"].get("strict_load_v", True),
                param_key,
            )

        self.net_g.train()
        self.net_v.train()

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("All losses are None. Please check.")

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        # optimizer g
        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, self.net_g.parameters(), **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt["optim_meta_g"].pop("type")
        self.optimizer_meta_g = self.get_optimizer(
            optim_type, self.net_v.parameters(), **train_opt["optim_meta_g"]
        )
        self.optimizers.append(self.optimizer_meta_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        if "meta_lq" in data:
            self.meta_lq = data["meta_lq"].to(self.device)
            self.meta_gt = data["meta_gt"].to(self.device)

    def determine_meta_weight(self, current_iter):

        if current_iter % 10 == 0:
            # update meta model
            loss_func = nn.L1Loss(reduction="none").to(self.device)

            inner_opt = torch.optim.SGD(
                self.net_g.parameters(), lr=1e-4
            )  # SGD for inner loop

            # Internal loop
            with higher.innerloop_ctx(self.net_g, inner_opt, copy_initial_weights=True) as (
                fnet,
                diffopt,
            ):
                # 1. Update meta model on training data
                output = fnet(self.lq)
                meta_train_loss = loss_func(output, self.gt)

                meta_train_loss = torch.mean(meta_train_loss, dim=(1, 2, 3), keepdim=False)  # 16 x 1
                meta_train_loss.backward(torch.ones_like(meta_train_loss))

                from torch.func import functional_call, vmap, grad
                params = {k: v.detach() for k, v in fnet.named_parameters()}
                buffers = {k: v.detach() for k, v in fnet.named_buffers()}

                
                ft_compute_grad = grad(compute_loss)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                ft_per_sample_grads = ft_compute_sample_grad(params, buffers, self.lq, self.gt) # might help to detach gradients

                out = torch.cat([output.data, self.gt.data], dim=1)
                w_xy_phi = self.net_v(out)  # w(x,y,phi)
                w_xy_phi = w_xy_phi.mean(dim=(1, 2, 3), keepdim=False)  # 16 x 1

                for i, (name, param) in enumerate(fnet.named_parameters()):
                    # Obtain the corresponding gradient
                    grad_for_param = ft_per_sample_grads[name]
                    # Reshape w_xy_phi for broadcasting for the current parameter's gradient
                    # Add singleton dimensions to match the gradient's dimensions, except for the batch dimension
                    reshaped_w_xy_phi = w_xy_phi.view(-1, *[1 for _ in range(grad_for_param.ndim - 1)])
                    # Weight the gradients by w_xy_phi (broadcasting happens here)
                    weighted_grad = grad_for_param * reshaped_w_xy_phi
                    # Sum the gradients over the batch dimension
                    summed_grad = weighted_grad.sum(0)
                    # Manually set the gradients
                    param.grad = summed_grad

                # Example of a manual update
                learning_rate = 1e-5  # very small learning rate
                for param in fnet.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad  # Basic gradient descent update

            # now fnet becomes theta_t
            J_e = loss_func(fnet(self.meta_lq), self.meta_gt)
            J_e = torch.mean(J_e)

            meta_val_loss_grad = torch.autograd.grad(J_e, (fnet.parameters()), create_graph=True)
            meta_val_loss_grad = {name: grad for (name, _), grad in zip(fnet.named_parameters(), meta_val_loss_grad)}

            for i, (name, param) in enumerate(fnet.named_parameters()):
                grad_theta_train = ft_per_sample_grads[name]
                grad_theta_val = meta_val_loss_grad[name][None]
                ndim = grad_theta_train.ndim

                dot_product = torch.sum(grad_theta_train * grad_theta_val, dim=list(1,ndim), keepdim=False)

            breakpoint()



            # check size
#            R_i = w_xy_phi * grad_term
#            phi_update_loss = torch.nn.functional.log(w_xy_phi) * R_i

            # External loop
            # 2. Update meta model on validation data
#            self.optimizer_meta_g.zero_grad()
#            meta_val_loss.backward()
#            self.optimizer_meta_g.step()

        with torch.no_grad():
            final_yhat = self.net_g(self.lq)
            out = torch.cat([final_yhat.data, self.gt.data], dim=1)
            w_new = self.net_v(out)

        return w_new

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if current_iter > -1:
            weights = self.determine_meta_weight(current_iter)
        else:
            weights = 1.0
        self.output = self.net_g(self.lq)

        # visualize weight and lq and gt images --> log images to tensorboard
        if self.weight_vis and current_iter % 1000 == 1:
            logger = get_root_logger()
            logger.info(f"Visualize weights at iteration {current_iter}")
            logger.info(f"weight is Mean: {weights.mean()}, STD: {weights.std()}")

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

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == "Adam":
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == "AdamW":
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == "Adamax":
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == "SGD":
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == "ASGD":
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == "RMSprop":
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == "Rprop":
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supported yet.")
        return optimizer
