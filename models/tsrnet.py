import torch
import torch.nn as nn
from .dgcnn_utils import STN, Backbone, SharedMLP1d
from .point_transformer_utils import PointTransformerBlock


class TSRNet(nn.Module):
    def __init__(self, args):
        super(TSRNet, self).__init__()

        if args.use_stn:
            self.stn = STN(args.k, args.norm)

        self.backbone = Backbone(args)

        self.fc_b = SharedMLP1d([args.emb_dims + args.n_edgeconvs_backbone*64+1, 256], args.norm)
        self.fc_d = SharedMLP1d([args.emb_dims + args.n_edgeconvs_backbone*64+1, 256], args.norm)

        self.b_fusion = PointTransformerBlock(256, 256, args.k)
        self.d_fusion = PointTransformerBlock(256, 256, args.k)

        self.b_out = nn.Sequential(SharedMLP1d([256, 128, 64], args.norm),
                                   nn.Dropout(args.dropout),
                                   nn.Conv1d(64, 3, kernel_size=1),)
        self.d_out = nn.Sequential(SharedMLP1d([256, 128, 64], args.norm),
                                   nn.Dropout(args.dropout),
                                   nn.Conv1d(64, 1, kernel_size=1),)

    def forward(self, x, bmap_pred, dmap_pred):
        device = x.device
        xyz = x.contiguous()  # B,3,N
        if hasattr(self, "stn"):
            t = self.stn(x.contiguous())
            x = torch.bmm(t, x)
        else:
            t = torch.ones((1, 1), device=device)

        x = self.backbone(x)

        feat_b = torch.cat((x, bmap_pred.unsqueeze(1)), dim=1)
        feat_d = torch.cat((x, dmap_pred.unsqueeze(1)), dim=1)

        # boundary map branch
        feat_b = self.fc_b(feat_b)
        feat_b = self.b_fusion(feat_b, xyz)
        b_out = self.b_out(feat_b)
        # distance map branch
        feat_d = self.fc_d(feat_d)
        feat_d = self.d_fusion(feat_d, xyz)
        d_out = self.d_out(feat_d)

        return b_out, d_out
