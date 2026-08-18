import time
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from .correct_method import Get_Method
from .util import unsqueeze_xdim
from utils.utils import *


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion:

    def __init__(self, betas, device):
            self.device = device

            # compute analytic std: eq 11
            std_fwd = np.sqrt(np.cumsum(betas))
            std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
            mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
            std_sb = np.sqrt(var)

            # tensorize everything
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.betas = to_torch(betas).to(device)
            self.std_fwd = to_torch(std_fwd).to(device)
            self.std_bwd = to_torch(std_bwd).to(device)
            self.std_sb = to_torch(std_sb).to(device)
            self.mu_x0 = to_torch(mu_x0).to(device)
            self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
            std_fwd = self.std_fwd[step]
            return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False, s=None):
            """Sample q(x_t | x_0, x_1), i.e. eq 11"""

            assert x0.shape == x1.shape
            batch, *xdim = x0.shape

            mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
            mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
            std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

            xt = mu_x0 * x0 + mu_x1 * x1
            if not ot_ode:
                if s is not None:
                    xt = xt + s * torch.conj(s).to("cuda") * std_sb * torch.randn_like(xt)
                else:
                    xt = xt + std_sb * torch.randn_like(xt)

            return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False, s=None):
            """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

            assert nprev < n
            std_n = self.std_fwd[n]
            std_nprev = self.std_fwd[nprev]
            std_delta = (std_n**2 - std_nprev**2).sqrt()

            mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

            xt_prev = mu_x0 * x0 + mu_xn * x_n
            if not ot_ode and nprev > 0:
                xt_prev = xt_prev + s * torch.conj(s) * var.sqrt() * torch.randn_like(
                    xt_prev
                )

            return xt_prev

    def ddim_inversion(self, nprev, n, x_n_1, delta):
            """
            delta = x_t - x0_pred = std_fwd[t] * eps_net
            inversion direction = -eps_net
            """

            std_n = self.std_fwd[n].to(x_n_1.device)
            std_nprev = self.std_fwd[nprev].to(x_n_1.device)

            coef_delta = (std_n ** 2 - std_nprev ** 2) / (std_nprev ** 2 + 1e-8)

            xn = x_n_1 - coef_delta * delta
            return xn

    def ddpm_sampling(
                self,
                idx,  # 添加
                steps,
                pred_x0_fn,
                x1,
                x2,  # 添加
                cg_weight,
                cg_iter,
                inver_step,
                n_cycles,
                mask=None,
                ot_ode=False,
                log_steps=None,
                verbose=True,
                s=None,
                correct_method=None,
        ):
            # correct method
            correct = (
                Get_Method(correct_method)
                if correct_method is not None
                else lambda pred_x0, mask, yk, x1, csm,cg_weight,cg_iter: pred_x0
            )

            xt = x1.detach().to(self.device)

            xs = []
            pred_x0s = []

            log_steps = log_steps or steps
            assert steps[0] == log_steps[0] == 0

            steps = steps[::-1]

            pair_steps = zip(steps[1:], steps[:-1])
            pair_steps = (
                tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1)
                if verbose
                else pair_steps
            )
            if mask is not None:
                print(len(mask))
                if len(mask) == 3:
                    mask, csm, yk = mask
                    print("yk read!")

                    isTest = False
                    startCorrect = False

                else:
                    mask, csm = mask
                csm = csm.to("cuda")
            # CGWEIGHT = 0.5
            # label_list = iter(np.linspace(0.1, 1.0, len(pair_steps)))
            for prev_step, step in pair_steps:
                assert prev_step < step, f"{prev_step=}, {step=}"

                # pred_x0 = pred_x0_fn(xt, step)

                # 假设你有一个需要计算梯度的函数
                def compute_gradient(xt, b, step, mask, pred_x0_fn):


                    # 在这个函数中，我们想要计算梯度，即使在torch.no_grad()的包裹下
                    b = fft2c_2d(r2c(b))
                    with torch.autograd.enable_grad():
                        xt.requires_grad_()
                        pred_x0 = pred_x0_fn(xt, step)

                        pred_b = mask.to("cuda") * fft2c_2d(r2c(pred_x0))
                        output = torch.nn.MSELoss()(
                            c2r(pred_b).type(torch.FloatTensor).to("cuda"),
                            c2r(b).type(torch.FloatTensor).to("cuda"),
                        )
                        grad = torch.autograd.grad(output, xt)[0]
                        print(output.mean())
                    xt = xt - 1e-5 * grad.type(torch.FloatTensor).to("cuda")
                    return pred_x0, xt

                # pred_x0, xt = compute_gradient(xt, yk, step, mask, pred_x0_fn)
                pred_x0 = pred_x0_fn(xt, step)

                # def normals(x):
                #     temp = r2c(x)
                #     minv = torch.std(temp)
                #     temp = temp / (1.5 * minv)
                #     return temp, minv

                # def back_normal(x, minv):
                #     return x * 1.5 * minv

                # corrector
                if mask is not None:


                    # def Giant_degrad(pred_x0, mask, yk, x1):
                    #     pred_x0, minv_p0 = normals(pred_x0)
                    #     temp, minv_x1 = normals(x1)
                    #     # # create Ax0
                    #     A_X0 = (
                    #         Emat_xyt_complex((pred_x0), False, None, (mask).to(pred_x0))
                    #     ).to("cuda")
                    #     coeef = 0.1  # single
                    #     # coeef = 2.5#multi
                    #     pred_x0 = (pred_x0) - coeef * ifft2c_2d(
                    #         A_X0 - Emat_xyt_complex(temp, False, None, 1)
                    #     )
                    #     pred_x0 = back_normal(pred_x0, minv_p0)
                    #     pred_x0 = c2r(pred_x0)
                    #     return pred_x0

                    # pass

                    # # pred_x0,minv_p0 = normals(pred_x0)
                    # # temp,minv_x1=normals(x1)
                    # # pred_x0 = (1-mask).to(pred_x0)*fft2c_2d((pred_x0))+fft2c_2d((temp))
                    # # pred_x0 = ifft2c_2d(pred_x0)
                    # # pred_x0 = back_normal(pred_x0,minv_p0)
                    # # pred_x0 = c2r(pred_x0)

                    # # # csm
                    # def CSM_method(pred_x0, mask, yk, x1):
                    #     pred_x0, minv_p0 = normals(pred_x0)
                    #     temp, minv_x1 = normals(yk if "yk" in locals() else x1)
                    #     # pred_x0 = (1 - mask.to(pred_x0)) * fft2c_2d(((pred_x0))) + fft2c_2d(
                    #     #     ((temp))
                    #     # )
                    #     # pred_x0 = ifft2c_2d(pred_x0)

                    #     pred_x0 = (1 - mask.to(pred_x0)) * fft2c_2d(
                    #         ((pred_x0)) * (csm).to(pred_x0)
                    #     ) + fft2c_2d(((temp)) * (csm).to(pred_x0))
                    #     pred_x0 = Emat_xyt_complex(pred_x0, True, csm.to(pred_x0), 1)
                    #     pred_x0 = back_normal(pred_x0, minv_p0)
                    #     pred_x0 = c2r(pred_x0)
                    #     return pred_x0

                    # # cg
                    # def CG_Method(pred_x0, mask, yk, x1):

                    #     pred_x0, minv_p0 = normals(pred_x0)
                    #     temp, minv_x1 = normals(yk if "yk" in locals() else x1)
                    #     # 参数待调整
                    #     # CGWEIGHT = next(label_list)
                    #     Aobj = Aclass(
                    #         c2r(csm),
                    #         mask.to("cuda"),
                    #         torch.tensor(0.5).cuda(),
                    #         num=True,
                    #     )
                    #     Rhs = c2r(temp)

                    #     pred_x0 = myCG(Aobj, Rhs, c2r(pred_x0), 5)
                    #     pred_x0 = back_normal(r2c(pred_x0), minv_p0)
                    #     pred_x0 = c2r(pred_x0).type(torch.FloatTensor).to("cuda")

                    #     return pred_x0
                    if startCorrect:
                        under = (
                            c2r(ifft2c_2d(mask.to("cuda") * fft2c_2d(r2c(xt))))
                            .type(torch.FloatTensor)
                            .to("cuda")
                        )
                        pred_underx0 = pred_x0_fn(under, step)

                        fourer_error_under = fft2c_2d(r2c(pred_underx0)) - fft2c_2d(
                            r2c(yk)
                        ).to("cuda")
                        pred_x0 = fft2c_2d(r2c(pred_x0)) - fourer_error_under
                        pred_x0 = c2r(ifft2c_2d(pred_x0)).type(torch.FloatTensor).to("cuda")

                        startCorrect = False
                        print(pred_x0.shape)
                        pass

                    if isTest:
                        print("log test!")
                        k0 = fft2c_2d(r2c(pred_x0))

                        minv = torch.std(k0)
                        k0 = k0 / (1.5 * minv)
                        yk = fft2c_2d(r2c(yk))
                        minv = torch.std(yk)
                        yk = yk / (1.5 * minv)

                        k = 48
                        start_index = (256 - k) // 2
                        end_index = start_index + k

                        # selective region mask
                        selection_mask = torch.zeros_like(mask)
                        selection_mask[
                            :, :, start_index:end_index, start_index:end_index
                        ] = 1

                        selection_mask = selection_mask * (~mask.bool())

                        yk[selection_mask.bool()] = k0[selection_mask.bool()]
                        yk = yk * (1.5 * minv)
                        yk = c2r(ifft2c_2d(yk))
                        # print(mask.shape)
                        # mask[:, :, start_index:end_index, start_index:end_index] = 1

                        isTest = False

                    # print("correct close")
                    pred_x0 = correct(pred_x0, mask, yk, x1, csm,cg_weight,cg_iter)

                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, s=s)

                # if mask is not None:
                #     xt_true = x1
                #     if not ot_ode:
                #         _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                #         std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                #         xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                #     xt = (1. - mask) * xt_true + mask * xt

                if prev_step in log_steps:
                    pred_x0s.append(pred_x0.detach().cpu())
                    xs.append(xt.detach().cpu())

            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def ddpm_inverse_sampling(
            self,
            idx,
            steps,
            pred_x0_fn,
            x1,#T1_img
            x2,#T2_img
            cg_weight,
            cg_iter,
            inver_step,
            n_cycles,
            mask=None, # [mask,t2_csm,欠采的t2_img]
            ot_ode=False,
            log_steps=None,
            verbose=True,
            s=None,
            correct_method=None,
            paired=None
        ):
            # correct method
            correct = (
                Get_Method(correct_method)
                if correct_method is not None
                else lambda pred_x0, mask, yk, x1, csm,cg_weight,cg_iter: pred_x0
            )

            xt = x1.detach().to(self.device)

            xs = []
            pred_x0s = []

            log_steps = log_steps or steps
            assert steps[0] == log_steps[0] == 0

            steps = steps[::-1]

            pair_steps_raw = zip(steps[1:], steps[:-1])
            pair_steps = (
                tqdm(pair_steps_raw, desc="DDPM sampling", total=len(steps) - 1)
                if verbose
                else pair_steps
            )
            if mask is not None:
                # print(len(mask))
                if len(mask) == 3:
                    mask, csm, yk = mask
                    # print("yk read!")

                    if paired:

                        def duplicated(data):
                            duplicated_tensor = torch.cat((data, data), dim=0).to(
                                data.device
                            )
                            return duplicated_tensor

                        mask = duplicated(mask)
                        csm = duplicated(csm)
                        yk = duplicated(yk)

                    isTest = False
                    startCorrect = False

                else:
                    yk = None
                    mask, csm = mask
                csm = csm.to("cuda")

            def apply_correct(pred):
                return correct(pred, mask, yk, x1, csm, cg_weight, cg_iter)
            # CGWEIGHT = 0.5

            # DDIM inversion
            t1 = time.time()
            for prev_step, step in pair_steps:

                # get initial x0
                pred_x0 = pred_x0_fn(xt, step)

                # from utils.utils import r2c
                # scio.savemat("src_87_x0.mat",{"x1_1":r2c(pred_x0).squeeze().detach().cpu().numpy()})
                # exit()
                pred_x0 = apply_correct(pred_x0)
                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, s=s)

            # pred_x0 = correct(pred_x0, mask, yk, x1, csm)

            # xt = Get_Method("CSM")(xt, mask, yk, x1, csm)
            # xt = pred_x0


            from i2sb.util import space_indices
            inver_step = int(inver_step)
            inversion_step = space_indices(inver_step, inver_step)
            # inversion_step = inversion_step[::1]
            inversion_step = inversion_step[:]
            # from utils.utils import r2c
            # scio.savemat("tar.mat",{"x0":r2c(xt).squeeze().detach().cpu().numpy()})
            # exit()
            xt_recon = xt.detach().clone()
            xt = xt_recon.clone()

            # xt = self.q_sample(inversion_step[0],xt,x1,ot_ode=True)
            start_i = 1
            xt = xt_recon.clone()

            for i in tqdm(range(len(inversion_step) - 1), desc="Inversion_process", ncols=100):
                step = inversion_step[i]
                step_next = inversion_step[i + 1]
                # x0 = pred_x0_fn(xt, step)

                x0 = pred_x0_fn(xt, step)
                delta = xt - x0

                # xt_next = self.ddim_inversion(
                #     step,
                #     step_next,
                #     xt,
                #     xt-x0
                # )
                xt_next = self.ddim_inversion(
                    step,
                    step_next,
                    xt,
                    delta
                )

                # mask = make_csm_support_mask(csm, thr=0.03).to(xt_next.device)

                # print("mask mean/max:", mask.mean().item(), mask.max().item())
                # xt_next = xt_next * mask
                # x1 = x1 * mask

                # if i in [0,99, 199,299, 399,499, 599, 799, len(inversion_step) - 2]:
                #     q_i = self.q_sample(step_next, xt_recon, x1, ot_ode=True)

                #     print(
                #         "i:", i,
                #         "step:", step,
                #         "next:", step_next,
                #         "mean|xt_next-q_i|:", torch.mean(torch.abs(xt_next - q_i)).item(),
                #         "mean|xt_next-T1|:", torch.mean(torch.abs(xt_next - x1)).item(),
                #         "mean|q_i-T1|:", torch.mean(torch.abs(q_i - x1)).item(),
                #         "mean|xt_next-T2recon|:", torch.mean(torch.abs(xt_next - xt_recon)).item()
                #     )
                #     mse_xt_t1 = torch.mean(torch.abs(xt_next - x1)).item()
                #     mse_xt_t1_curve.append(mse_xt_t1)
                #     step_curve.append(step_next)
                #     if step_next in [0, 100, 200, 400, 600, 800, 999]:
                #         # b: 原始 T1 / guidance image
                #         save_b_bhat_error_png(
                #             b=x1,
                #             b_hat=xt_next,
                #             step=step
                #         )


                xt = xt_next




                # xt_next = self.ddim_inversion(
                #     step,
                #     step_next,
                #     xt,
                #     xt-x0,
                #     x0
                # )

                # ===== 每隔 100 个 inversion step 保存一次 update 后的结果 =====
            #     if (step_next % 50 == 0) or (i == len(inversion_step) - 2):
            #         save_xt[f"xt_{step_next}"] = r2c(xt_next).squeeze().detach().cpu().numpy()

            #     xt = xt_next

            # plot_nmse_curve_first_200_steps(
            #     step_curve=step_curve,
            #     mse_xt_t1_curve=mse_xt_t1_curve,
            #     filename="nmse_curve_first_200_steps.png",
            #     max_step=200,
            # )

            b_hat_cot1 = xt
            # # b_hat_cot1 = lowfreq_replace(xt, x1, beta=0.6, kernel_size=21, sigma=5.0)
            # # b_hat_cot1 = match_intensity_stats(xt, x1)
            # xt = b_hat_cot1
            # print("inversion_step[0] =", inversion_step[0])
            # print("inversion_step[-1] =", inversion_step[-1])
            # # ===== 保存路径 =====
            # print("xt的尺寸是：", xt.shape)
            # ===== 保存路径 =====
            # 保存到 .mat
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"brain_{idx}.mat")
            # scio.savemat(save_path, save_xt)
            # plt.figure()
            # plt.plot(step_curve, mse_xt_t1_curve)
            # plt.xlabel("Inversion step")
            # plt.ylabel("mean|xt_next - T1|")
            # plt.title("Inversion MSE to T1")
            # plt.grid(True)
            # save_curve_path = os.path.join(curve_dir, f"mse_xt_next_T1_curve_{idx}.png")
            # plt.savefig(save_curve_path, dpi=300, bbox_inches="tight")
            # plt.close()
            # print(f"Saved curve to: {save_curve_path}")
            # print(f"Saved inversion steps to: {save_path}")

                # andersion
                # def anderson_function(xt):
                #     x0 = pred_x0_fn(xt,inversion_step[i+1])
                #     xt= self.ddim_inversion(inversion_step[i],inversion_step[i+1],xtminus,xt-x0)
                #     return xt

                # xt,_ = anderson(anderson_function,xt,)

            # pair_steps = (
            #     tqdm(pair_steps_raw, desc="DDPM sampling", total=len(steps) - 1)
            #     if verbose
            #     else pair_steps
            # )
            from copy import deepcopy


            # scio.savemat("refine_87.mat",{"x1_1":r2c(xt).squeeze().detach().cpu().numpy()})
            # scio.savemat("src_87.mat",{"x1":r2c(x1).squeeze().detach().cpu().numpy()})
            # exit()


            pair_steps_raw = zip(steps[1:], steps[:-1])

            for j in range(0):
                # xt = (xt + x1) / 2
                # DDIM inversion
                for prev_step, step in deepcopy(pair_steps_raw):
                    # get initial x0
                    pred_x0 = pred_x0_fn(xt, step)
                    pred_x0 = apply_correct(pred_x0)
                    xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, s=s)

            # for i in range(800):
            #     xt = correct(xt, mask, yk, x1, csm)

            i = 0
            # xt = (xt + x1) / 2
            for prev_step, step in deepcopy(pair_steps_raw):

                assert prev_step < step, f"{prev_step=}, {step=}"

                # pred_x0 = pred_x0_fn(xt, step)

                # # 假设你有一个需要计算梯度的函数
                # def compute_gradient(xt, b, step, mask, pred_x0_fn):
                #     from utils.utils import fft2c_2d, r2c, c2r
                #
                #     # 在这个函数中，我们想要计算梯度，即使在torch.no_grad()的包裹下
                #     b = fft2c_2d(r2c(b))
                #     with torch.autograd.enable_grad():
                #         xt.requires_grad_()
                #         pred_x0 = pred_x0_fn(xt, step)
                #
                #         pred_b = mask.to("cuda") * fft2c_2d(r2c(pred_x0))
                #         output = torch.nn.MSELoss()(
                #             c2r(pred_b).type(torch.FloatTensor).to("cuda"),
                #             c2r(b).type(torch.FloatTensor).to("cuda"),
                #         )
                #         grad = torch.autograd.grad(output, xt)[0]
                #         # print(output.mean())
                #     xt = xt - 1e-5 * grad.type(torch.FloatTensor).to("cuda")
                #     return pred_x0, xt

                pred_x0 = pred_x0_fn(xt, step)

                # scio.savemat("refine_87__x0.mat",{"x1_1":r2c(pred_x0).squeeze().detach().cpu().numpy()})
                # exit()

                final_correct = lambda x: x
                # corrector
                if mask is not None:
                    # from utils.utils import (
                    #     c2r,
                    #     Emat_xyt_complex,
                    #     fft2c_2d,
                    #     r2c,
                    #     ifft2c,
                    #     ifft2c_2d,
                    #     fft2c,
                    # )

                    if startCorrect:
                        under = (
                            c2r(ifft2c_2d(mask.to("cuda") * fft2c_2d(r2c(xt))))
                            .type(torch.FloatTensor)
                            .to("cuda")
                        )
                        pred_underx0 = pred_x0_fn(under, step)

                        fourer_error_under = fft2c_2d(r2c(pred_underx0)) - fft2c_2d(
                            r2c(yk)
                        ).to("cuda")
                        pred_x0 = fft2c_2d(r2c(pred_x0)) - fourer_error_under
                        pred_x0 = c2r(ifft2c_2d(pred_x0)).type(torch.FloatTensor).to("cuda")

                        startCorrect = False
                        print(pred_x0.shape)
                        pass

                    if isTest:
                        print("log test!")
                        k0 = fft2c_2d(r2c(pred_x0))

                        minv = torch.std(k0)
                        k0 = k0 / (1.5 * minv)
                        yk = fft2c_2d(r2c(yk))
                        minv = torch.std(yk)
                        yk = yk / (1.5 * minv)

                        k = 48
                        start_index = (256 - k) // 2
                        end_index = start_index + k

                        # selective region mask
                        selection_mask = torch.zeros_like(mask)
                        selection_mask[
                            :, :, start_index:end_index, start_index:end_index
                        ] = 1

                        selection_mask = selection_mask * (~mask.bool())

                        yk[selection_mask.bool()] = k0[selection_mask.bool()]
                        yk = yk * (1.5 * minv)
                        yk = c2r(ifft2c_2d(yk))
                        # print(mask.shape)
                        # mask[:, :, start_index:end_index, start_index:end_index] = 1

                        isTest = False
                        #第二次采样的cg
                    pred_x0 = apply_correct(pred_x0)

                    # if i == len(steps[1:]):
                    #     pred_x0 = Get_Method("CSM")(pred_x0, mask, yk, x1, csm)

                    i = i + 1
                    from functools import partial

                    if i == len(steps[1:]):
                        final_correct = partial(
                            Get_Method("CSM"), mask=mask, yk=yk, x1=x1, csm=csm
                        )

                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, s=s)

                # if i == len(steps[1:]):
                #     xt = final_correct(xt)

                # if mask is not None:
                #     xt_true = x1
                #     if not ot_ode:
                #         _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                #         std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                #         xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                #     xt = (1. - mask) * xt_true + mask * xt

                if prev_step in log_steps:
                    pred_x0s.append(pred_x0.detach().cpu())
                    xs.append(xt.detach().cpu())

            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            t2 = time.time()

            # print("time",t2 - t1)

            return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def sampling_method(self, name):
        methods = {"ddpm": self.ddpm_sampling, "inversion": self.ddpm_inverse_sampling}
        try:
            return methods[name.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported sampling method: {name}") from error
