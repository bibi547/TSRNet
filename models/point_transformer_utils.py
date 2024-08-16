from models.pointnet_utils import index_points
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from models.dgcnn_utils import knn


class PointTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_points),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc_delta = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.w_qs = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.w_ks = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.w_vs = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.k = k

    # xyz: b x 3 x n, map: b x 1 x n, features: b x f x n
    def forward(self, feats, pos):

        knn_idx = knn(pos, self.k)
        knn_pos = index_points(pos.permute(0,2,1), knn_idx)

        pre = feats
        x = self.fc1(feats)
        q = self.w_qs(x)  # b x f x n
        k = index_points(self.w_ks(x).permute(0,2,1), knn_idx).permute(0, 3, 2, 1)  # b x f x k x n
        v = index_points(self.w_vs(x).permute(0,2,1), knn_idx).permute(0, 3, 2, 1)  # b x f x k x n

        pos = pos.permute(0, 2, 1)[:, :, None] - knn_pos
        pos = pos.permute(0, 3, 2, 1)
        pos_enc = self.fc_delta(pos)  # b x f x k x n

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x f x k x n

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)  # b x f x n
        res = self.fc2(res) + pre  # b x f x n

        return res




