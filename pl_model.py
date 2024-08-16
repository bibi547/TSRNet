import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.tsrnet import TSRNet
from data.teeth3ds_dataset import Teeth3DS


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TSRNet(args)

        self.criterion_bmap = F.cross_entropy  # or focal loss
        self.criterion_dmap = F.l1_loss  # or F.mse_loss

    def forward(self, x, bmap, dmap):
        return self.net(x, bmap, dmap)

    def infer(self, x, bmap, dmap):
        b_out, d_out = self(x, bmap, dmap)
        return b_out, d_out

    def training_step(self, batch, _):
        x, bmap_gt, bmap, dmap_gt, dmap = batch
        b_out, d_out = self(x, bmap, dmap)

        loss_b = self.criterion_bmap(b_out.squeeze(), bmap_gt)
        loss_d = self.criterion_dmap(d_out.squeeze(), dmap_gt)
        loss = loss_b + loss_d * 5
        self.log('loss', loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        x, bmap_gt, bmap, dmap_gt, dmap = batch
        b_out, d_out = self(x, bmap, dmap)

        loss_b = self.criterion_bmap(b_out.squeeze(), bmap_gt)
        loss_d = self.criterion_dmap(d_out.squeeze(), dmap_gt)
        loss = loss_b + loss_d * 5
        self.log('val_loss', loss, True)

    def test_step(self, batch, _):
        x, bmap_gt, bmap, dmap_gt, dmap = batch
        b_out, d_out = self(x, bmap, dmap)

        loss_b = self.criterion_bmap(b_out.squeeze(), bmap_gt)
        loss_d = self.criterion_dmap(d_out.squeeze(), dmap_gt)
        loss = loss_b + loss_d * 5
        self.log('test_loss', loss, True)

    def configure_optimizers(self):
        args = self.hparams.args
        steps_per_epoch = (len(self.train_dataloader()) + args.gpus - 1) // args.gpus  # for multi-gpus
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs, steps_per_epoch=steps_per_epoch)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)
