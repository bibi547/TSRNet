# TSRNet: A Dual-stream Network for Refining 3D Tooth Segmentation

## Run

### 1. Mesh Simplification

Simplify original tooth meshes to about 10,000 facets. 
Mesh simplification can be achieved using functions from [Open3D](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html) or through implementations by [MeshCNN](https://github.com/ranahanocka/MeshCNN/blob/master/scripts/dataprep/blender_process.py), which utilize the bpy library.

### 2. Coarse Segmentation Methods

[DGCNN](https://github.com/WangYueFt/dgcnn), TeethGNN, or others.

### 3. Extracting Geodesic Distance Maps

[./scripts/geodesic_distance.py](https://github.com/bibi547/TSRNet/tree/master/scripts)

Geodesic distance maps are extracted during the data preprocessing phase. 
Based on the ground truth and the coarse segmentation results, extract the segmentation boundaries (vertices). 
Then, compute the shortest geodesic distance from mesh vertices to the segmentation boundaries and save the results as '.txt' files.

### 4. Config

Modify the [config](https://github.com/bibi547/TSRNet/blob/master/config/teeth3ds_cfg.yaml) and [dataset](https://github.com/bibi547/TSRNet/blob/master/data/teeth3ds_dataset.py) based on the path and filename.

### 5. Requirement

...

### 6. Train

```
python train.py
```


### 7. Test

```
python test.py
```

## Citation

```
@article{jin2024tsrnet,
  title={TSRNet: A Dual-stream Network for Refining 3D Tooth Segmentation},
  author={Jin, Hairong and Shen, Yuefan and Lou, Jianwen and Zhou, Kun and Zheng, Youyi},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement

[MeshCNN](https://github.com/ranahanocka/MeshCNN/blob/master/scripts/dataprep/blender_process.py)

[DGCNN](https://github.com/WangYueFt/dgcnn)

[Point Transformer](https://github.com/qq456cvb/Point-Transformers)

[Teeth3DS dataset](https://github.com/abenhamadou/3dteethseg22_challenge)


