import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    MeanSquaredError,
)


def Evaluate_ssimAndpsnr(label, recon):

    if torch.is_tensor(label):
        pass
    else:
        label = torch.from_numpy(label)
        recon = torch.from_numpy(recon)
    label = torch.abs(label).type(torch.FloatTensor)
    recon = torch.abs(recon).type(torch.FloatTensor)
    label = torch.unsqueeze(label, 0)
    label = torch.unsqueeze(label, 0)
    recon = torch.unsqueeze(recon, 0)
    recon = torch.unsqueeze(recon, 0)

    if len(recon.size()) != 4:

        print(recon.size())
        x, y = recon.size()[-1], recon.size()[-2]
        recon = recon.view(1, 1, x, y)
        label = label.view(1, 1, x, y)
    # >0.1
    label = torch.abs(label)
    mask1 = label > 0.1 * torch.max(label)
    recon = recon * mask1
    label = label * mask1

    ssim = StructuralSimilarityIndexMeasure()
    res = ssim(recon, label)

    psnr = PeakSignalNoiseRatio()
    psnrs = psnr(recon, label)

    recon = recon / torch.max(recon)
    label = label / torch.max(label)
    err = torch.abs(recon - label)
    nmse = torch.linalg.norm(err.view(-1)) ** 2 / torch.linalg.norm(label.view(-1)) ** 2
    # print(a_mse)

    # print(res)
    # print(psnrs)
    # print(nmse)
    return res, psnrs, nmse


# Evaluate_ssimAndpsnr(label,recon)

# label = torch.from_numpy(label)
# recon = torch.from_numpy(recon)
# label = torch.abs(label).type(torch.FloatTensor)
# recon = torch.abs(recon).type(torch.FloatTensor)
# label = torch.unsqueeze(label, 0)
# label = torch.unsqueeze(label, 0)
# recon = torch.unsqueeze(recon, 0)
# recon = torch.unsqueeze(recon, 0)

# ssim = StructuralSimilarityIndexMeasure()
# res = ssim(recon, label)
# print(res)
