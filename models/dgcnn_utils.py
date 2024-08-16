import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.long()

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def get_normlayer1d(norm):
    if norm == 'instance':
        normlayer = nn.InstanceNorm1d
    elif norm == 'batch':
        normlayer = nn.BatchNorm1d
    else:
        assert 0, "not implemented"
    return normlayer


def get_normlayer2d(norm):
    if norm == 'instance':
        normlayer = nn.InstanceNorm2d
    elif norm == 'batch':
        normlayer = nn.BatchNorm2d
    else:
        assert 0, "not implemented"
    return normlayer


def MLP(channels, norm):
    if norm=='batch':
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i-1], channels[i], bias=False),
                          nn.BatchNorm1d(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))])
    elif norm=='instance':
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i-1], channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))])
    else:
        assert 0


class SharedMLP2d(nn.Module):
    def __init__(self, channels, norm):
        super(SharedMLP2d, self).__init__()
        normlayer = get_normlayer2d(norm)

        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(channels[i-1], channels[i], kernel_size=1, bias=False),
                          normlayer(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))
        ])

    def forward(self, x):
        return self.conv(x)


class SharedMLP1d(nn.Module):
    def __init__(self, channels, norm):
        super(SharedMLP1d, self).__init__()
        normlayer = get_normlayer1d(norm)

        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=False),
                          normlayer(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))
        ])

    def forward(self, x):
        return self.conv(x)


class EdgeConv(nn.Module):
    def __init__(self, channels, k, norm):
        super(EdgeConv, self).__init__()

        self.k = k
        self.smlp = SharedMLP2d(channels, norm)

    def forward(self, x, idx=None):
        """
        :param x: (N_batch, n_features, n_faces)
        :param idx: (N_batch, n_
        :return: output: (N_batch, n_features_out, n_faces)
        """
        x = get_graph_feature(x, self.k, idx)       # (N, channels[0], num_points, k)
        x = self.smlp(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class STN(nn.Module):
    """
    Spatial transformer network
    """
    def __init__(self, k, norm):
        super(STN, self).__init__()

        self.conv1 = EdgeConv([3*2, 64, 128], k, norm)
        self.smlp = SharedMLP1d([128, 1024], norm)

        self.mlp = MLP([1024, 512, 256], norm)

        self.transform = nn.Linear(256, 3*3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)     # (batch_size, 3, num_points) -> (batch_size, 128, num_points)
        x = self.smlp(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = self.mlp(x)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3).contiguous()            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class Backbone(nn.Module):
    def __init__(self, args):
        """
        @param args: k, int, the number of neighbors.
                     dynamic, bool, if using dynamic or not.
                     input_channels, int
                     n_edgeconvs_backbone, int, the number of EdgeConvs in the backbone
                     emb_dims, int, the dim of embedding features
                     norm, str, "instance" or "batch"
                     global_pool_backbone, str, "avg" or "max"
        """
        super(Backbone, self).__init__()
        self.k = args.k
        self.dynamic = args.dynamic

        channel = args.input_channels
        self.convs = nn.ModuleList()
        for _ in range(args.n_edgeconvs_backbone):
            self.convs.append(EdgeConv([channel*2, 64, 64], args.k, args.norm))
            channel = 64

        self.smlp = SharedMLP1d([args.n_edgeconvs_backbone*64, args.emb_dims], args.norm)

        # global pooling
        if args.global_pool_backbone == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif args.global_pool_backbone == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            assert 0

    def forward(self, x):
        """
        @param x: with shape(batch, input_channels, n), the first 3 dims of input_channels are xyz
        @param idx: the initial idx with shape(batch, n, k), if None then compute by x
        @return: output with `emb_dims+n_edgeconvs*64` channels
        """
        # idx = knn(x[:, :3, :].contiguous(), self.k)      # calc idx according to xyz
        idx = knn(x.contiguous(), self.k)
        x = self.convs[0](x, idx)    # the first edgeconv

        xs = [x]
        for i in range(1, len(self.convs)):
            x = self.convs[i](x, None if self.dynamic else idx)
            xs.append(x)
        x = torch.cat(xs, dim=1)

        x = self.smlp(x)
        x_pool = self.pool(x)
        x_pool = x_pool.expand(x_pool.shape[0], x_pool.shape[1], x.shape[2])  # torch.repeat will slow the backward process in pytorch 1.5

        x = torch.cat((x_pool, *xs), dim=1)
        return x


