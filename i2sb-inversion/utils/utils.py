import numpy as np
import torch
import torch.fft as FFT
import torch.nn.functional as F


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img

def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)

def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)

def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)

def fft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x

def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x

def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)
    return x

def ifft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))
    return x

def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x

def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)
    return x

def Emat_xyt(b, inv, csm, mask, num=True):
    if csm is None:
        value = r2c(b)
        value = (ifft2c_2d(value * mask) if value.ndim == 4 else ifft2c(value * mask)) if inv else ((fft2c_2d(value) if value.ndim == 4 else fft2c(value)) * mask)
        return c2r(value)

    csm = r2c(csm)
    if inv:
        value = r2c(b) * mask
        value = ifft2c_2d(value) if value.ndim == 4 else ifft2c(value)
        if num:
            value = torch.sum(value * torch.conj(csm), 1, keepdim=True)
    else:
        value = r2c(b)
        if num:
            value = value * csm
        value = (fft2c_2d(value) if value.ndim == 4 else fft2c(value)) * mask
    return c2r(value)


def Emat_xyt_complex(b, inv, csm, mask,num=True,pad=False):
    if csm is None:
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
                if pad is not False:
                    x = pad_to_256x256(x)
            else:
                x = ifft2c(x)
                if pad is not False:
                    x = pad_to_256x256(x)
            # print(x.size())
            x = x.squeeze()
            if num:
                x = x*torch.conj(csm)
                x = torch.sum(x, 1)
                x = torch.unsqueeze(x, 1)
            else:
                x = torch.unsqueeze(x, 0)

        else:
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b

    return x

def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x

def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x

def pad_to_256x256(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """
    Center-pad the last two dims (H, W) to (256, 256).
    Works for x with shape (..., H, W). Keeps dtype/device.
    """
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims (..., H, W), got {x.shape}")

    H, W = x.shape[-2:]
    target_h, target_w = 256, 256
    if H == 256 and W == 256:
        return x

    if H > target_h or W > target_w:
        raise ValueError(
            f"Input spatial size {(H, W)} larger than {(target_h, target_w)}; cannot pad."
        )
    pad_h = target_h - H
    pad_w = target_w - W

    pad = (
        pad_w // 2,                # left
        pad_w - pad_w // 2,        # right
        pad_h // 2,                # top
        pad_h - pad_h // 2,        # bottom
    )

    # 对 complex tensor，F.pad 的 value 只能是 0
    pad_value = 0 if torch.is_complex(x) else value

    return F.pad(x, pad, mode="constant", value=pad_value)

def crop_to_210x240(x: torch.Tensor) -> torch.Tensor:
    """
    Center-crop the last two dims (H, W) to (210, 240).
    Works for x with shape (..., H, W). Keeps dtype/device.
    """
    if x.dim() < 2:
        raise ValueError(f"Expected tensor with at least 2 dims (..., H, W), got {x.shape}")

    H, W = x.shape[-2], x.shape[-1]
    target_h, target_w = 210, 240
    if H < target_h or W < target_w:
        raise ValueError(f"Input spatial size {(H, W)} smaller than {(target_h, target_w)}; cannot crop.")

    start_y = (H - target_h) // 2
    start_x = (W - target_w) // 2
    return x[..., start_y:start_y + target_h, start_x:start_x + target_w]
