import numpy as np
from collections import defaultdict
from sklearn import metrics


def get_contour_points(vertex_faces, labels):
    contour_points = []
    for i,adj_faces in enumerate(vertex_faces):
        adj_faces = np.setdiff1d(adj_faces, -1)
        adj_labels = labels[adj_faces]
        if len(set(adj_labels)) > 1:
            contour_points.append(i)
    return contour_points


def get_iou(ori_idx, tar_idx):

    inter = np.intersect1d(ori_idx, tar_idx)
    union = np.union1d(ori_idx, tar_idx)
    iou = len(inter)/len(union)

    return iou


def get_faces_per_classes(labels, tar_labels):
    faces_dict = defaultdict(list)
    faces_dict = faces_dict.fromkeys(tar_labels, [])
    for i, l in enumerate(labels):
        if l in tar_labels:
            faces = faces_dict[l][:]
            faces.append(i)
            faces_dict[l] = faces
    return faces_dict


def get_tooth_iou(preds, labels):
    miou = 0.0
    tooth_num = 0.0
    tooth_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    tooth_preds = get_faces_per_classes(preds, np.array(tooth_idx))
    tooth_labels = get_faces_per_classes(labels, np.array(tooth_idx))
    for l in tooth_labels:
        if len(tooth_labels[l]) == 0 or len(tooth_preds[l]) == 0:
            continue
        labels = tooth_labels[l]
        preds = tooth_preds[l]
        iou = get_iou(preds, labels)
        miou += iou
        tooth_num += 1
    miou = miou/tooth_num
    return miou


def get_metrics(preds, gt, pred_contour, label_contour, vertex_faces):
    acc = metrics.accuracy_score(list(preds.squeeze()), list(gt.squeeze()))
    contour_iou = get_iou(label_contour, pred_contour)
    tooth_iou = get_tooth_iou(preds, gt)
    return acc, tooth_iou, contour_iou


def get_refined_labels(fs, labels, preds, vertex_vertices, vertex_faces, b_out, d_out):
    pred_contour = get_contour_points(vertex_faces, preds)
    label_contour = get_contour_points(vertex_faces, labels)
    acc, tiou, biou = get_metrics(preds, labels, pred_contour, label_contour, vertex_faces)
    all_acc = []
    all_tiou = []
    all_biou = []
    all_acc.append(acc)
    all_tiou.append(tiou)
    all_biou.append(biou)
    print(acc, tiou, biou)
    # iteration 10
    for i in range(10):
        for p in pred_contour:
            neighbors_vs = np.array(vertex_vertices[p])
            neighbors_vs = np.setdiff1d(neighbors_vs, pred_contour)
            neighbors_vs = np.append(p, neighbors_vs)
            neighbors_bmap = b_out[neighbors_vs]

            if neighbors_bmap[0] == 0:
                continue

            # 计算每个face的距离值
            neighbors_fs = np.setdiff1d(vertex_faces[p], -1)
            neighbors_dists = np.zeros(neighbors_fs.shape[0])
            for j, face in enumerate(neighbors_fs):
                dists = d_out[fs[face]]
                face_dist = np.mean(dists)
                neighbors_dists[j] = face_dist
            tar_face = neighbors_fs[np.argmax(neighbors_dists)]
            tar_label = preds[tar_face]
            preds[neighbors_fs] = tar_label
        # get new contour
        pred_contour = get_contour_points(vertex_faces, preds)
        acc, tiou, biou = get_metrics(preds, labels, pred_contour, label_contour, vertex_faces)
        all_acc.append(acc)
        all_tiou.append(tiou)
        all_biou.append(biou)
        print(acc, tiou, biou)
    return all_acc, all_tiou, all_biou, preds




