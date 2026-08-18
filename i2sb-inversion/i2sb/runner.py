# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import time

import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage

from . import util
from .network import Image256Net
from .diffusion import Diffusion
from utils.utils import *
import i2sb.correct_method


def build_optimizer_sched(opt, net, log):
    optim_dict = {"lr": opt.lr, "weight_decay": opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    # optimizer = RMSprop(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, "gamma": opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if (
            sched is not None
            and "sched" in checkpoint.keys()
            and checkpoint["sched"] is not None
        ):
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()


def single_channel_Emat(t1_k0, t1_csm, t2_k0, t2_csm, multi):
    label = Emat_xyt_complex(t2_k0, True, t2_csm, 1, multi == False)  # 1x1x320x320

    label = label/label.abs().max()

    clean_img = c2r(label).type(torch.FloatTensor).to("cuda")

    corrupt_img = (Emat_xyt_complex(t1_k0, True, t1_csm, multi == False))
    corrupt_img = corrupt_img/corrupt_img.abs().max()
    corrupt_img = (
        c2r(corrupt_img)
        .type(torch.FloatTensor)
        .to("cuda")
    )


    y = torch.tensor(1).to("cuda")
    return clean_img, corrupt_img, y


class Runner(object):
    def __init__(self, opt, log, save_opt=True, multi_coil=False, topk=None):
        super(Runner, self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))
        # opt.beta_max = 1
        betas = make_beta_schedule(
            n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval
        )
        print("beta_max为：", opt.beta_max)
        betas = np.concatenate(
            [betas[: opt.interval // 2], np.flip(betas[: opt.interval // 2])]
        )
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = (
            torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device)
            * opt.interval
        )

        # build net
        # print(opt.image_size)
        # from torchinfo import summary

        self.net = Image256Net(
            log,
            noise_levels=noise_levels,
            use_fp16=opt.use_fp16,
            cond=opt.cond_x1,
            inchannel=opt.channel,
            image_size=opt.image_size,
            topk=topk,
        )
        x = torch.randn(1, 2, 256, 256)          # 图像输入
        t = torch.tensor([1])                 # 时间步输入（可能是 int，可能是 embedding）

        # summary(self.net.diffusion_model, input_size=(x, t))
        # exit()

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.multi = multi_coil
        self.dataset = opt.dataset

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint["net"], strict=False)
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """Eq 12"""
        # compute label taken part in loss-calculation
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method=None):
        if opt.corrupt != "T1T2" or opt.dataset != "T1T2":
            raise ValueError("This minimal release supports only --corrupt T1T2 --dataset T1T2")
        t1_kspace, t1_maps, t2_kspace, t2_maps = next(loader)
        clean_img, corrupt_img, y = single_channel_Emat(
            t1_kspace, t1_maps, t2_kspace, t2_maps, self.multi
        )
        mask = None
        y = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        # if mask is not None:
        #     mask = mask.detach().to(opt.device)
        #     x1 = (1.0 - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise:  # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape
        #T2/T1
        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader = util.setup_loader(val_dataset, opt.microbatch)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        # add mask

        # 在 for it 循环外，先定义
        warmup_it = 10
        measure_it = 100  # 每100个it统计一次平均耗时
        t_list = []

        for it in range(opt.num_itr):
            if opt.global_rank == 0:
                torch.cuda.synchronize(opt.device)
                it_t0 = time.time()

            optimizer.zero_grad()

            # todo:>>> 1) 在这里 reset（建议只测少数几个 it，避免很慢）
            do_mem_profile = (opt.global_rank == 0) and (it in [0, 1, 2])  # 你也可以换成 it==10 等
            if do_mem_profile:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(opt.device)
                torch.cuda.synchronize(opt.device)

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(
                    opt, train_loader, corrupt_method
                )

                # start = 0
                # stop = 1000
                # num_integers = 5

                # step = (stop - start) // (num_integers - 1)

                # result = list(range(start, stop + 1, step))
                # result = [0, 200, 400, 600, 800, 999]
                # for i in result:
                #     # print((torch.ones((x0.shape[0],)) * i).int().dtype)

                #     xt = self.diffusion.q_sample(
                #         i,
                #         x0,
                #         x1,
                #         ot_ode=opt.ot_ode,
                #         s=y,
                #     )
                #     xt = r2c(xt)
                #     save_mat("figs/per_results", xt, "xt", i, False)
                # exit()

                # result = [0, 100, 200, 300, 400, 500, 600, 700, 800, 999]
                # xt_all = []
                # for i in result:
                #     # print((torch.ones((x0.shape[0],)) * i).int().dtype)

                #     xt = self.diffusion.q_sample(
                #         i,
                #         x0,
                #         x1,
                #         ot_ode=opt.ot_ode,
                #         s=y,
                #     )
                #     xt = r2c(xt)
                #     xt_all.append(xt)
                # stacked_xt = torch.stack(xt_all, dim=0)
                # save_mat("figs/per_results", stacked_xt, "xt_all", i, False)
                # exit()



                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],), device=opt.device)

                # print(torch.isnan(x0).any())
                # print(torch.isnan(x1).any())
                # log.info("start from random noise")
                # x1 = torch.randn_like(x1).to(x1.device)

                #对应Algorithm 1 Training第三步
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode, s=y)
                xt = pad_to_256x256(xt)
                x0 = pad_to_256x256(x0)
                label = self.compute_label(step, x0, xt)
                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                loss = F.mse_loss(pred, label)

                loss.backward()

            # 检查梯度中是否存在NaN值
            nan_found = False
            for param in net.parameters():
                if torch.isnan(param.grad).any():
                    nan_found = True
                    break

            if nan_found:
                print("梯度中存在NaN值！")
                # print(torch.isnan(xt).any())
                # print(torch.isnan(label).any())
                # print(torch.isnan(pred).any())

            else:
                pass
                # print("梯度中没有NaN值。")

            # 如果没有NaN值，则更新参数
            if not nan_found:
                optimizer.step()
                ema.update()
                # >>> 2) 在这里读取峰值（放在 step 之后）
            if do_mem_profile:
                torch.cuda.synchronize(opt.device)
                peak_train_gb = torch.cuda.max_memory_allocated(opt.device) / 1024 ** 3
                peak_train_reserved_gb = torch.cuda.max_memory_reserved(opt.device) / 1024 ** 3
                print(
                    f"[it={it}] peak_train_alloc={peak_train_gb:.3f} GB | peak_train_reserved={peak_train_reserved_gb:.3f} GB")

            # optimizer.step()
            # ema.update()
            if sched is not None:
                sched.step()

            # >>> 在这里统计每个 it 的耗时（rank0即可）
            if opt.global_rank == 0:
                torch.cuda.synchronize(opt.device)
                dt = time.time() - it_t0

                # 跳过前warmup_it个it
                if it >= warmup_it:
                    t_list.append(dt)

                # 每满100个it，打印一次平均耗时
                if len(t_list) == measure_it:
                    avg_it_time = sum(t_list) / measure_it
                    print(f"[AVG] recent {measure_it} iters: {avg_it_time:.4f} s/it")
                    t_list = []

            # -------- logging --------
            log.info(
                "train_it {}/{} | lr:{} | loss:{}".format(
                    1 + it,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]["lr"]),
                    "{:+.4f}".format(loss.item()),
                )
            )
            if it % 10 == 0:
                self.writer.add_scalar(it, "loss", loss.detach())

            if it % 10000 == 0:
                if opt.global_rank == 0:
                    torch.save(
                        {
                            "net": self.net.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "sched": sched.state_dict() if sched is not None else sched,
                        },
                        opt.ckpt_path / ("latest_%d.pt" % int(it / 10000)),
                    )
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # if it == 10 or it % 3000 == 0:  # 0, 0.5k, 3k, 6k 9k
            #     net.eval()
            #     self.evaluation(opt, it, val_loader, corrupt_method)
            #     # self.evaluation(opt, it, val_loader, corrupt_method)
            #     net.train()
        self.writer.close()

    def ddpm_sampling(
        self,
        opt,
            idx,
        x1,#T1_img
        x2,#T2_img
        cg_weight,
        cg_iter,
        inver_step,
        n_cycles,
        y=None,
        mask=None,#[mask,t2_csm,欠采的t2_img]
        cond=None,
        clip_denoise=False,
        nfe=None,
        log_count=10,
        verbose=True,
        s=None,
        correct_method=None,
        sampling_method = 'ddpm'
    ):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe + 1)

        # create log steps
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None:
            cond = cond.to(opt.device)
        # if mask is not None:
        #     mask = mask.to(opt.device)
        #     x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def crop_to_input(img, size):
                h, w = size
                top = (img.shape[-2] - h) // 2
                left = (img.shape[-1] - w) // 2
                return img[..., top:top + h, left:left + w]

            def pred_eps_fn(xt, step):
                input_size = xt.shape[-2:]
                step_t = torch.full(
                    (xt.shape[0],), step, device=opt.device, dtype=torch.long
                )

                xt_pad = pad_to_256x256(xt)
                eps = self.net(xt_pad, step_t, cond=None)

                return crop_to_input(eps, input_size)

            def pred_x0_fn(xt, step):
                input_size = xt.shape[-2:]
                step = torch.full(
                    (xt.shape[0],), step, device=opt.device, dtype=torch.long
                )
                # from torchinfo import summary
                # summary(self.net, input_data=(xt, step))
                # exit()
                xt = pad_to_256x256(xt)
                out = self.net(xt, step, cond=cond)
                pred_x0 = self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)
                return crop_to_input(pred_x0, input_size)
                # return out

            def pred_x0_fn_adv(xt, step):
                step = torch.full(
                    (xt.shape[0],), step, device=opt.device, dtype=torch.long
                )
                latent_z = torch.randn(1, 100, device=xt.device)
                out = self.net(xt, step, latent_z)
                return out

            sampling_method = self.diffusion.sampling_method(sampling_method)

            xs, pred_x0 = sampling_method(
                idx,
                steps,
                pred_x0_fn,
                x1,#T1_img
                x2,
                cg_weight=cg_weight,
                cg_iter=cg_iter,
                inver_step=inver_step,
                n_cycles=n_cycles,
                mask=mask,#[mask,t2_csm,欠采的t2_img]
                ot_ode=opt.ot_ode,
                log_steps=log_steps,
                verbose=verbose,
                s=s,
                correct_method=correct_method,
            )


        b, *xdim = x1.shape
        # print(xs.shape)
        # print(pred_x0.shape)
        assert xs.shape == pred_x0.shape
        return xs, pred_x0
