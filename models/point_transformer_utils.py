from models.pointnet_utils import index_points, square_distance, PointNetFeaturePropagation, PointNetSetAbstraction
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from models.dgcnn_utils import knn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.npoints, self.nneighbor = args.num_points, args.k
        self.n_c = 17
        self.d_points = 3
        self.nblocks = 4
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, 512, self.nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(self.nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(self.npoints // 4 ** (i + 1), self.nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, 512, self.nneighbor))

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


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

        # d_idx = knn(feats, self.k)  # dynamic k idx
        s_idx = knn(pos, self.k)  # static k idx
        # knn_idx = torch.cat([d_idx, s_idx], dim=2)
        knn_idx = s_idx
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




