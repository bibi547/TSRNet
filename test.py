import os
import tqdm
import click
import torch
import trimesh
import numpy as np

from pl_model import LitModel
from data.teeth3ds_dataset import Teeth3DS
from utils.metrics import get_refined_labels


@click.command()
@click.option('--gpus', default=1)
def run(gpus):

    weight_file = 'E:/code/tooth_refine/TSRNet/runs/teeth3ds/upper.ckpt'
    # write_path = 'F:/dataset/Teeth3DS/results/dgcnn/upper_refined/'
    model = LitModel.load_from_checkpoint(weight_file).cuda()
    args = model.hparams.args
    model = model.eval()
    dataset = Teeth3DS(args, args.test_file, False)

    all_acc = []
    all_miou = []
    all_biou = []

    for i in tqdm.tqdm(range(len(dataset))):

        off_file = dataset.files[i][0]
        print(off_file)
        dmap_pred_file = dataset.files[i][2]
        mesh = trimesh.load_mesh(off_file)
        vs, fs = np.array(mesh.vertices), np.array(mesh.faces)
        vertex_vertices, vertex_faces = mesh.vertex_neighbors, mesh.vertex_faces
        dmap_pred = np.loadtxt(dmap_pred_file, dtype=np.float64)
        bmap_pred = np.where(dmap_pred > 0, 1, np.where(dmap_pred < 0, -1, 0)) + 1

        vs = torch.tensor(vs.T, dtype=torch.float32).unsqueeze(0).cuda()
        dmap_pred = torch.tensor(dmap_pred, dtype=torch.float32).unsqueeze(0).cuda()
        bmap_pred = torch.tensor(bmap_pred, dtype=torch.float32).unsqueeze(0).cuda()

        b_out, d_out = model.infer(vs, bmap_pred, dmap_pred)
        b_out = b_out[0].softmax(0).cpu().detach().T.numpy()  # (n, 3)
        d_out = d_out.cpu().detach().numpy().squeeze()
        b_out = b_out.argmax(-1) - 1
        d_out = np.abs(d_out)

        filename = os.path.basename(off_file).split('.')[0]
        category = filename.split('_')[1]  # upper/lower
        fn = filename.split('_')[0]
        gt_file = os.path.join(dataset.root_dir, category, fn, filename + '_re.txt')
        gt = np.loadtxt(gt_file, dtype=np.int64)
        pred_file = os.path.join('F:/dataset/Teeth3DS/results/dgcnn/', category, filename + '.txt')# dataset.pred_dir
        preds = np.loadtxt(pred_file, dtype=np.int64)

        accs, tious, bious, preds = get_refined_labels(fs, gt, preds, vertex_vertices, vertex_faces, b_out, d_out)

        # write_file = os.path.join(write_path, filename + '.txt')
        # np.savetxt(write_file, preds, fmt='%d', delimiter=',')
        all_acc.append(accs)
        all_miou.append(tious)
        all_biou.append(bious)

    all_acc = np.array(all_acc)
    all_miou = np.array(all_miou)
    all_biou = np.array(all_biou)

    acc_0 = np.mean(all_acc[:, 0])
    acc_9 = np.mean(all_acc[:, 9])
    miou_0 = np.mean(all_miou[:, 0])
    miou_9 = np.mean(all_miou[:, 9])
    biou_0 = np.mean(all_biou[:, 0])
    biou_9 = np.mean(all_biou[:, 9])

    print('acc 0:', acc_0)
    print('acc 9:', acc_9)
    print('miou 0:', miou_0)
    print('miou 9:', miou_9)
    print('biou 0:', biou_0)
    print('biou 9:', biou_9)


if __name__ == "__main__":
    run()
