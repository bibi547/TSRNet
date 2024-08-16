# TSRNet

## TSRNet: A Dual-stream Network for Refining 3D Tooth Segmentation

### 1. Mesh Simplification

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

### 5. Train

```
python train.py
```


### 6. Evaluation

