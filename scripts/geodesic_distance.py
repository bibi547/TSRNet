import numpy as np
import trimesh
from sklearn.manifold import Isomap


def get_contour_points(faces, labels):
    contour_points = []
    for i,adj_faces in enumerate(faces):
        adj_faces = np.setdiff1d(adj_faces, -1)
        adj_labels = labels[adj_faces]
        if len(set(adj_labels)) > 1:
            contour_points.append(i)
    return contour_points


def get_boundary_map(vertices, faces, labels, contour):

    p_sdf = np.zeros(vertices.shape[0])
    for i, face in enumerate(faces):
        if labels[i] == 0:
            p_sdf[face] = -1
        else:
            p_sdf[face] = 1
    for p in contour:
        p_sdf[p] = 0
    return p_sdf


def get_distance_map(vertices, faces, labels, contour):

    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    data_2d = isomap.fit_transform(X=vertices)
    geo_distance_metrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample]
    contour_distance = geo_distance_metrix[:, contour]
    min_distance = np.min(contour_distance, axis=1)

    p_sdf = get_boundary_map(vertices, faces, labels, contour)
    p_sdf = p_sdf * min_distance

    # normalize to [-1,1]
    max = np.max(p_sdf)
    min = -np.min(p_sdf)
    for i, d in enumerate(p_sdf):
        if d > 0:
            p_sdf[i] /= max
        if d < 0:
            p_sdf[i] /= min

    return p_sdf


import os
if __name__ == '__main__':

    shape_path = ''  # simplified mesh
    pred_path = ''  # ground truth or coarse segmentation results (face-wise)
    write_path = ''  # for distance maps

    sub_files = os.listdir(shape_path)
    for filename in sub_files:
        off_file = os.path.join(shape_path, filename, filename + '.off')
        pred_file = os.path.join(pred_path, filename, filename + '.txt')
        write_file = os.path.join(write_path, filename + '.txt')
        labels = np.loadtxt(pred_file, dtype='int64')

        mesh = trimesh.load(off_file)
        vertices = np.array(mesh.vertices, dtype='float32')
        faces = np.array(mesh.faces)
        vertex_faces = np.array(mesh.vertex_faces)
        contour = get_contour_points(vertex_faces, labels)
        p_sdf = get_distance_map(vertices, faces, labels, contour)
        p_sdf = np.array(p_sdf, dtype='float32')
        np.savetxt(write_file, p_sdf, fmt='%f', delimiter=',')

        print(filename, ' written!')



