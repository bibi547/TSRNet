import os
import numpy as np
import random

import torch
import trimesh
from torch.utils.data import Dataset


class Teeth3DS(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        # args
        self.args = args
        self.root_dir = args.root_dir
        self.pred_dir = args.pred_dir
        self.num_points = args.num_points
        self.augmentation = args.augmentation if train else False
        # files
        self.files = []
        with open(os.path.join(args.split_dir, split_file)) as f:
            for line in f:
                filename = line.strip().split('_')[0]
                category = line.strip().split('_')[1]  # upper/lower

                shape_root = os.path.join(self.root_dir, category, filename)
                off_file = os.path.join(shape_root, f'{line.strip()}_sim.off')

                dmap_gt_root = os.path.join(self.root_dir, category + '_dmap')
                dmap_gt_file = os.path.join(dmap_gt_root, f'{line.strip()}_sim.txt')

                dmap_pred_root = os.path.join(self.pred_dir, category + '_dmap')
                dmap_pred_file = os.path.join(dmap_pred_root, f'{line.strip()}_sim.txt')

                if os.path.exists(off_file) and os.path.exists(dmap_gt_file) and os.path.exists(dmap_pred_file):
                    self.files.append((off_file, dmap_gt_file, dmap_pred_file))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        off_file, dmap_gt_file, dmap_pred_file = self.files[idx]

        mesh = trimesh.load(off_file)
        vs, fs = np.array(mesh.vertices), np.array(mesh.faces)
        dmap_gt = np.loadtxt(dmap_gt_file, dtype=np.float64)
        dmap_pred = np.loadtxt(dmap_pred_file, dtype=np.float64)
        # bmap [-1, 0, 1] + 1 : gingiva boundary tooth
        bmap_gt = np.where(dmap_gt > 0, 1, np.where(dmap_gt < 0, -1, 0))+1
        bmap_pred = np.where(dmap_pred > 0, 1, np.where(dmap_pred < 0, -1, 0))+1

        # augmentation
        # if self.augmentation:
        #     vs, _ = augment(vs, fs)
        # sample
        # _, fids = trimesh.sample.sample_surface_even(mesh, self.num_points)
        sampl_ids = np.random.permutation(len(vs))[:self.num_points]
        vs = vs[sampl_ids]
        bmap_gt = bmap_gt[sampl_ids]
        bmap_coarse = bmap_pred[sampl_ids]
        dmap_gt = dmap_gt[sampl_ids]
        dmap_coarse = dmap_pred[sampl_ids]

        return torch.tensor(vs.T, dtype=torch.float32), torch.tensor(bmap_gt, dtype=torch.long), \
            torch.tensor(bmap_coarse, dtype=torch.long), torch.tensor(dmap_gt, dtype=torch.float32), \
            torch.tensor(dmap_coarse, dtype=torch.float32)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.root_dir = 'F:/dataset/Teeth3DS/data'
            self.split_dir = 'F:/dataset/Teeth3DS/split'
            self.pred_dir = 'F:/dataset/Teeth3DS/results/dgcnn'
            self.num_points = 4096
            self.augmentation = True

    data = Teeth3DS(Args(), 'training_upper.txt', True)
    i = 0
    for vs, bg, bc, dg, dc in data:
        print(vs.shape)
        print(bg.shape)
        print(bc.shape)
        print(dg.shape)
        print(dc.shape)
        i += 1

    print(i)