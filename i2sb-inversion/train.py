import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Process

import distributed_util
from dataset.dataset import get_dataset
from i2sb import Runner
from logger import Logger


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    set_seed(opt.seed + opt.global_rank)
    dataset_class = get_dataset(opt)
    train_dataset = dataset_class(opt, mode="training")
    val_dataset = dataset_class(opt, mode="sample")
    runner = Runner(opt, log)
    runner.train(opt, train_dataset, val_dataset, corrupt_method=None)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal I2SB T1-to-T2 MRI training")
    parser.add_argument("--name", default="brain_run")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--n-gpu-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-proc-node", type=int, default=1)
    parser.add_argument("--master-address", default="localhost")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--num-itr", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--log-writer", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--dataset", default="T1T2")
    parser.add_argument("--corrupt", default="T1T2")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--normalize_type", default="img_std")
    parser.add_argument("--normalize_coeff", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=1e-4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--beta-max", type=float, default=0.3)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--lr-gamma", type=float, default=0.99)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--cond-x1", action="store_true")
    parser.add_argument("--add-x1-noise", action="store_true")
    parser.add_argument("--ot-ode", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.batch_size % args.microbatch:
        raise ValueError("--batch-size must be divisible by --microbatch")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args.device = "cuda"
    args.distributed = args.n_gpu_per_node > 1
    args.global_size = args.num_proc_node * args.n_gpu_per_node
    args.ckpt_path = ROOT / "results" / args.name
    args.ckpt_path.mkdir(parents=True, exist_ok=True)
    args.load = args.resume

    if args.distributed:
        processes = []
        for local_rank in range(args.n_gpu_per_node):
            process_opt = copy.deepcopy(args)
            process_opt.local_rank = local_rank
            process_opt.global_rank = local_rank + args.node_rank * args.n_gpu_per_node
            process = Process(target=distributed_util.init_processes, args=(process_opt.global_rank, args.global_size, main, process_opt))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    else:
        torch.cuda.set_device(0)
        args.local_rank = args.global_rank = 0
        distributed_util.init_processes(0, 1, main, args)
