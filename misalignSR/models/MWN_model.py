import torch
import torch.nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from misalignSR.models.LRE_model import LRESRModel
import higher
import numpy as np
from misalignSR.losses.basic_loss import GradientLoss
# from torch.func import functional_call, vmap, grad
import functorch
from misalignSR.losses.basic_loss import GWLoss
import torchopt


@MODEL_REGISTRY.register()
class MWNSRModel(LRESRModel):
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

        # optimizer for meta-weight-net
        lr = self.opt['train']['optim_g'].get("lr", 1e-4)
        self.g_model_optimizer = torchopt.MetaSGD(self.net_g, lr=lr)  # use it for MWN
        self.v_model_optimizer = torchopt.MetaSGD(self.net_v, lr=lr)  # use it for MWN

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

        net_g_initial_params = [p.clone() for p in self.net_g.parameters()]
        net_v_initial_params = [p.clone() for p in self.net_v.parameters()]

        if self.meta_loss_type == 'GW':
            loss_func = GWLoss().to(self.device)
        else:
            loss_func = torch.nn.L1Loss().to(self.device)

        fmodel, theta = functorch.make_functional(self.net_g)
        wmodel, phi = functorch.make_functional(self.net_v)

        learning_rate = 1e-4
        model_optimizer = torchopt.sgd(learning_rate)
        model_opt_state = model_optimizer.init(theta)

        def compute_loss(params, sample, target):
            if sample.ndim == 3:
                sample = sample.unsqueeze(0)  # prepend batch dimension for processing
                target = target.unsqueeze(0)
            prediction = fmodel(params, sample)  # prediction
            loss = loss_func(prediction, target)
            return loss

        # Now we perform the inner loop
        weight_train_input = torch.cat((self.lq, self.gt), dim=1)
        meta_weights = wmodel(phi, weight_train_input)

        # compute per-sample gradients for training batch
        per_sample_loss = torch.func.vmap(compute_loss, in_dims=(None, 0, 0))(theta, self.lq, self.gt)
        per_sample_grads = torch.func.vmap(torch.func.grad(
            compute_loss), in_dims=(None, 0, 0),)(theta, self.lq, self.gt)

        inner_loss = torch.sum(per_sample_loss * meta_weights)
        inner_grad = torch.autograd.grad(inner_loss, theta, create_graph=True)

        updates, opt_state = model_optimizer.update(inner_grad, model_opt_state, inplace=True)
        theta_t = torchopt.apply_updates(theta, updates, inplace=True)

        # Now we perform the outer loop
        val_loss = compute_loss(theta_t, self.meta_lq, self.meta_gt)
        val_grad = torch.autograd.grad(val_loss, theta_t, create_graph=True) # J_e(Dv, theta_t)

        grad_dot_product = sum([torch.sum(tg * vg, dim=tuple(range(1, tg.ndim)))
                               for (tg, vg) in zip(per_sample_grads, val_grad)])

        alpha = 1e-2
        with torch.no_grad():
            R_i = (grad_dot_product + alpha * torch.log(meta_weights + 1e-8) + alpha) * meta_weights

        def compute_gradients_phi(R_i, meta_weights, phi):
            # First we compute d_phi_gi

            nb = len(meta_weights)
            eta = 1e-3

            d_phi_gi = []

            for g_i in meta_weights:
                d_phi_gi.append(torch.autograd.grad(torch.log(g_i), phi, create_graph=True))

            for r_i, _d_phi_gi in zip(R_i, d_phi_gi):
                for t_d_phi_gi, _phi in zip(_d_phi_gi, phi):
                    tmp = _phi.data + t_d_phi_gi.data * r_i * eta / nb
                    _phi.data.copy_(tmp)

            return phi

        #phi_initial_params = [p.clone() for p in phi]

        phi = compute_gradients_phi(R_i, meta_weights, phi)

        '''
        for initial, updated in zip(phi_initial_params, phi):
            if not torch.equal(initial, updated):
                print("Phi Parameter updated")
                break
        else:
            print("Phi: No parameter was updated")
        '''

        for param, param_phi in zip(self.net_v.parameters(), phi):
            param.data.copy_(param_phi.data)

        '''
        for initial, updated in zip(net_g_initial_params, self.net_g.parameters()):
            if not torch.equal(initial, updated):
                print("Net G Parameter updated")
                break
        else:
            print("Net G: No parameter was updated")

        for initial, updated in zip(net_v_initial_params, self.net_v.parameters()):
            if not torch.equal(initial, updated):
                print("Net V Parameter updated")
                break
        else:
            print("Net V: No parameter was updated")
        '''
        return

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if current_iter > -1:
            self.determine_meta_weight(current_iter)
            weight_train_input = torch.cat((self.lq, self.gt), dim=1)
            with torch.no_grad():
                weights = self.net_v(weight_train_input)
            weights = weights * len(self.lq)
            #print(weights)
        else:
            weights = 1.0
        self.output = self.net_g(self.lq)
        if current_iter % 100 == 0:
            logger = get_root_logger()
            logger.info(f"Visualize weights at iteration {current_iter}")
            logger.info(f"weight is Mean: {weights.mean()}, STD: {weights.std()}")

        # visualize weight and lq and gt images --> log images to tensorboard
        if self.weight_vis and current_iter % 100 == 1:
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
            l_pix = self.cri_pix(weights[:,None,None,None] * self.output, weights[:,None,None,None] * self.gt)
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
