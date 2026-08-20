![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg)

<img width="1362" height="746" alt="image" src="https://github.com/user-attachments/assets/64e334d3-aa90-49dc-9116-5cf83cdbc287" />


# I2SB-Inversion

This is the codebase for **I2SB-Inversion**, including training and sampling for brain and knee MRI reconstruction.

# Installation

```bash
pip install -r requirements.txt
```
# Pretrained Checkpoints and Test Data

The pretrained checkpoints and test samples are available on [[Google Drive](https://drive.google.com/drive/my-drive).](https://drive.google.com/drive/folders/1TwGgkjUwvv3mm34fEP1aeaQgNpPXB9gB?usp=sharing)

After downloading, please place the files in the corresponding checkpoints/ and test_data/ directories.

# Folder Structure

```text
i2sb-inversion/
├── train.py
├── sample.py
├── dataset/
├── guided_diffusion/
├── i2sb/
├── utils/
├── checkpoints/
│   ├── brain/latest.pt
│   └── knee/latest.pt
├── masks/
│   ├── brain/ellip_acc11.38_center25.mat
│   └── knee/ellip_acc11.38_center26.mat
└── test_data/
    ├── brain/
    │   ├── T1/brain_slice_T1.h5
    │   └── T2/brain_slice_T2.h5
    └── knee/
        ├── T1/knee_slice_T1.h5
        └── T2/knee_slice_T2.h5
```

# Testing

## Knee

```bash
python sample.py \
  --ckpt='./checkpoints/knee' \
  --n-gpu-per-node=1 \
  --dataset-dir='' \
  --batch-size=1 \
  --nfe=80 \
  --inver_step=1000 \
  --dataset='T1T2' \
  --channel=2 \
  --mask='./masks/knee/ellip_acc11.38_center26.mat' \
  --mask_type='ellip' \
  --acc='11.38' \
  --acs=26 \
  --cg_weight=7e-5 \
  --cg_iter=7 \
  --normalize_type='img_std' \
  --sampling_method='inversion' \
  --correct=CG \
  --n_cycles=1 \
  --output-dir='./outputs/knee_result'
```

Use `--sampling_method='ddpm'` for **I2SB-Recon**.

## Brain

```bash
python sample.py \
  --ckpt='./checkpoints/brain' \
  --n-gpu-per-node=1 \
  --dataset-dir='' \
  --batch-size=1 \
  --nfe=80 \
  --inver_step=1000 \
  --dataset='T1T2' \
  --channel=2 \
  --mask='./masks/brain/ellip_acc11.38_center25.mat' \
  --mask_type='ellip' \
  --acc='11.38' \
  --acs=25 \
  --cg_weight=7e-5 \
  --cg_iter=7 \
  --normalize_type='img_std' \
  --sampling_method='inversion' \
  --correct=CG \
  --n_cycles=1 \
  --output-dir='./outputs/brain_result'
```

Use `--sampling_method='ddpm'` for **I2SB-Recon**.

# Training

```bash
python train.py \
  --name='brain_run' \
  --n-gpu-per-node=1 \
  --dataset-dir='/path/to/DATASET_ROOT' \
  --batch-size=16 \
  --log-dir='logs' \
  --microbatch=1 \
  --corrupt='T1T2' \
  --log-writer='tensorboard' \
  --dataset='T1T2' \
  --channel=2 \
  --image-size=256 \
  --lr=1e-5 \
  --interval=1000 \
  --normalize_type='img_std'
```
