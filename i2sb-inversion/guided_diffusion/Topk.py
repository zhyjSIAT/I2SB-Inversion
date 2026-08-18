import torch
import torch.nn as nn


class SparseHWModule(nn.Module):
    def __init__(self, tau, topk, device="cuda"):
        super(SparseHWModule, self).__init__()
        self.tau = tau
        self.topk = topk
        self.device = device

    def forward(self, x):
        n, c, h, w = x.shape

        if self.topk == 1:
            return x
        x_reshape = x.view(n, c, h * w)
        topk_keep_num = int(max(1, self.topk * h * w))
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(self.device)

        sparse_x = mask * x_reshape
        sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
        if self.tau == 1.0:
            return sparse_x.view(n, c, h, w)

        tau_x = x * torch.FloatTensor([1.0 - self.tau]).to(self.device)
        return (
            sparse_x.view(n, c, h, w) * torch.FloatTensor([self.tau]).to(self.device)
            + tau_x
        )
