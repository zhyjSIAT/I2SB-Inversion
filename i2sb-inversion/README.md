
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
    ├── brain/T1/brain_slice_T1.h5 + T2/brain_slice_T2.h5
    └── knee/T1/knee_slice_T1.h5 + T2/knee_slice_T2.h5
```

```bash
pip install -r requirements.txt
```

```bash
git lfs install
git lfs track "checkpoints/**/*.pt"
```

## test

### knee 

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

### brain

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

training：

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
