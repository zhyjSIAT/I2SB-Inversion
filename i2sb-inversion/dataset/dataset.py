from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import Emat_xyt_complex, FFT2c, IFFT2c, normalize_complex


_DATASETS = {}


def pair_key(path):
    stem = path.stem
    for suffix in ("_T1", "_T2", "-T1", "-T2"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def register_dataset(cls=None, *, name=None):
    def register(target):
        _DATASETS[name or target.__name__] = target
        return target

    return register(cls) if cls is not None else register


def get_dataset(config):
    return _DATASETS[config.dataset]


@register_dataset(name="T1T2")
class T1T2MRIDataSet(Dataset):
    """Paired multi-coil T1/T2 HDF5 dataset.

    Expected layout for a user dataset:
      DATASET_ROOT/train/T1/*.h5 and DATASET_ROOT/train/T2/*.h5
      DATASET_ROOT/test/T1/*.h5  and DATASET_ROOT/test/T2/*.h5

    Each paired file must have the same name and contain `kspace` plus either
    `s_maps` or `csm`, with the slice dimension first.
    """

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        root = self._resolve_root(config, mode)
        self.t1_dir = root / "T1"
        self.t2_dir = root / "T2"
        self.items = []

        t2_files = {pair_key(path): path for path in self.t2_dir.glob("*.h5")}
        for t1_file in sorted(self.t1_dir.glob("*.h5")):
            key = pair_key(t1_file)
            if key not in t2_files:
                raise FileNotFoundError(f"Missing paired T2 file for {t1_file.name}")
            t2_file = t2_files[key]
            with h5py.File(t1_file, "r") as data:
                n_slices = data["kspace"].shape[0]
            self.items.extend((t1_file, t2_file, index) for index in range(n_slices))

        if not self.items:
            raise RuntimeError(f"No paired HDF5 slices found under {root}")

    @staticmethod
    def _resolve_root(config, mode):
        dataset_dir = str(config.dataset_dir)
        if dataset_dir and dataset_dir != ".":
            root = Path(dataset_dir).expanduser().resolve()
            split = "train" if mode == "training" else "test"
            return root / split if (root / split).is_dir() else root

        if mode == "training":
            raise ValueError("Training requires --dataset-dir pointing to paired train/T1 and train/T2 data.")

        project_root = Path(__file__).resolve().parents[1]
        anatomy = "knee" if "knee" in str(config.ckpt).lower() else "brain"
        return project_root / "test_data" / anatomy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        t1_file, t2_file, slice_index = self.items[index]
        output = []
        for path in (t1_file, t2_file):
            with h5py.File(path, "r") as data:
                map_key = "s_maps" if "s_maps" in data else "csm"
                kspace = np.nan_to_num(np.asarray(data["kspace"][slice_index]), nan=0.0)
                maps = np.nan_to_num(np.asarray(data[map_key][slice_index]), nan=0.0)

            if self.config.normalize_type == "minmax":
                k = torch.from_numpy(kspace[None])
                m = torch.from_numpy(maps[None])
                image = Emat_xyt_complex(k, True, m, 1)
                image = self.config.normalize_coeff * normalize_complex(image)
                kspace = Emat_xyt_complex(image, False, m, 1).squeeze(0).numpy()
            elif self.config.normalize_type == "std":
                scale = self.config.normalize_coeff * np.std(kspace)
                kspace = kspace / (scale if scale else 1.0)
            elif self.config.normalize_type == "img_std":
                image = IFFT2c(kspace[None])
                scale = np.max(np.abs(image))
                image = image / (scale if scale else 1.0)
                kspace = np.squeeze(FFT2c(image), 0)

            output.extend((np.asarray(kspace), np.asarray(maps)))

        return tuple(output)
