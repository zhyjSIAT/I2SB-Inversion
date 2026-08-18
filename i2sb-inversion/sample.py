import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torchvision.utils as tv_utils
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset

import distributed_util as dist_util
from dataset.dataset import get_dataset
from i2sb import Runner
from i2sb import ckpt_util
from logger import Logger
from utils.ssim import Evaluate_ssimAndpsnr
from utils.utils import Emat_xyt_complex, c2r, crop_to_210x240, normalize_complex, r2c


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_from_kspace(kspace, maps):
    image = Emat_xyt_complex(kspace, True, maps, 1, True)
    image = image / image.abs().max()
    return c2r(image).float().cuda()


def prepare_batch(batch, mask_path):
    t1_kspace, t1_maps, t2_kspace, t2_maps = batch
    t1_image = image_from_kspace(t1_kspace, t1_maps)
    t2_image = image_from_kspace(t2_kspace, t2_maps)

    mask = sio.loadmat(mask_path)["mask"].astype(np.complex128)
    mask = torch.from_numpy(mask[None, None])
    if tuple(mask.shape[-2:]) != tuple(t2_kspace.shape[-2:]):
        raise ValueError(f"Mask {tuple(mask.shape[-2:])} does not match data {tuple(t2_kspace.shape[-2:])}")

    undersampled_t2 = c2r(
        Emat_xyt_complex(t2_kspace * mask, True, t2_maps, 1, True)
    ).float().cuda()
    correction = [mask, t2_maps, undersampled_t2]
    return t1_image, t2_image, correction, t2_maps


def magnitude_normalized(image):
    return torch.abs(normalize_complex(r2c(image)))


@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    ckpt_path = Path(opt.ckpt).expanduser().resolve()
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, ckpt_path)
    ckpt_opt.dataset = "T1T2"

    dataset = get_dataset(opt)(opt, mode="sample")
    indices = np.arange(len(dataset))[opt.global_rank::opt.global_size].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=opt.batch_size, shuffle=False, num_workers=0)
    runner = Runner(ckpt_opt, log, save_opt=False)

    anatomy = "knee" if "knee" in ckpt_path.name.lower() else "brain"
    if opt.mask:
        mask_path = Path(opt.mask).expanduser().resolve()
    else:
        candidates = sorted((ROOT / "masks" / anatomy).glob("*.mat"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one mask under masks/{anatomy}, found {len(candidates)}; pass --mask")
        mask_path = candidates[0]
    output_dir = Path(opt.output_dir or ROOT / "outputs" / f"{anatomy}_{opt.sampling_method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for index, batch in enumerate(loader, start=1):
        t1_image, t2_image, correction, t2_maps = prepare_batch(batch, mask_path)
        samples, _ = runner.ddpm_sampling(
            ckpt_opt,
            index - 1,
            t1_image,
            t2_image,
            cg_weight=opt.cg_weight,
            cg_iter=opt.cg_iter,
            inver_step=opt.inver_step,
            n_cycles=opt.n_cycles,
            mask=correction,
            cond=None,
            nfe=opt.nfe,
            verbose=True,
            s=torch.tensor(1, device="cuda"),
            correct_method=opt.correct,
            sampling_method=opt.sampling_method,
        )

        recon = magnitude_normalized(samples[:, 0])
        label = magnitude_normalized(t2_image)
        recon = crop_to_210x240(recon)
        label = crop_to_210x240(label)

        sio.savemat(
            output_dir / f"recon_{index}.mat",
            {"recon": recon.squeeze().cpu().numpy(), "label": label.squeeze().cpu().numpy()},
        )
        tv_utils.save_image(recon, output_dir / f"recon_{index}.png", normalize=True)
        ssim, psnr, nmse = Evaluate_ssimAndpsnr(label.squeeze(), recon.squeeze())
        metrics.append((float(ssim), float(psnr), float(nmse)))
        log.info(f"slice={index} ssim={ssim:.6f} psnr={psnr:.6f} nmse={nmse:.6f}")

    values = np.asarray(metrics)
    log.info(f"mean_ssim={values[:, 0].mean():.6f}")
    log.info(f"mean_psnr={values[:, 1].mean():.6f}")
    log.info(f"mean_nmse={values[:, 2].mean():.6f}")
    if dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal I2SB T1-to-T2 MRI reconstruction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpu-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-proc-node", type=int, default=1)
    parser.add_argument("--master-address", default="localhost")
    parser.add_argument("--dataset-dir", type=Path, default=Path(""))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nfe", type=int, default=80)
    parser.add_argument("--inver_step", type=float, default=1000)
    parser.add_argument("--dataset", default="T1T2")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--mask_type", default="ellip")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--acc", required=True)
    parser.add_argument("--acs", required=True)
    parser.add_argument("--cg_weight", type=float, default=7e-5)
    parser.add_argument("--cg_iter", type=float, default=7)
    parser.add_argument("--normalize_type", default="img_std")
    parser.add_argument("--normalize_coeff", type=float, default=1.5)
    parser.add_argument("--sampling_method", choices=("ddpm", "inversion"), default="inversion")
    parser.add_argument("--correct", default="CG")
    parser.add_argument("--n_cycles", type=int, default=1)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_seed(args.seed)
    opt = args
    opt.device = "cuda"
    opt.distributed = opt.n_gpu_per_node > 1
    opt.global_size = opt.num_proc_node * opt.n_gpu_per_node

    if opt.distributed:
        processes = []
        for local_rank in range(opt.n_gpu_per_node):
            process_opt = copy.deepcopy(opt)
            process_opt.local_rank = local_rank
            process_opt.global_rank = local_rank + opt.node_rank * opt.n_gpu_per_node
            process = Process(target=dist_util.init_processes, args=(process_opt.global_rank, opt.global_size, main, process_opt))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    else:
        torch.cuda.set_device(0)
        opt.local_rank = opt.global_rank = 0
        dist_util.init_processes(0, 1, main, opt)
