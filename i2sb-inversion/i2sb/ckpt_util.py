import pickle
from pathlib import Path


I2SB_IMG256_UNCOND_PKL = "256x256_diffusion_uncond_fixedsigma.pkl"
I2SB_IMG256_COND_PKL = "256x256_diffusion_cond_fixedsigma.pkl"


def config_cartesian(image_size, topk=None):
    config = {
        "image_size": image_size,
        "num_channels": 256,
        "num_res_blocks": 2,
        "channel_mult": "",
        "learn_sigma": False,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": "32,16,8",
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
    }
    if topk:
        config.update(topk=0.1, topk_layer=[3, 4, 5])
    return config


def build_ckpt_option(runtime_opt, log, ckpt_path):
    ckpt_path = Path(ckpt_path)
    options_path = ckpt_path / "options.pkl"
    with options_path.open("rb") as handle:
        options = pickle.load(handle)
    options.use_fp16 = runtime_opt.use_fp16
    options.device = runtime_opt.device
    checkpoint = ckpt_path / "latest.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_path / "latest_5.pt"
    options.load = checkpoint
    log.info(f"Loaded options from {options_path}!")
    return options
