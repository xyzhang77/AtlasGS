<div align="center">

# **Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians**

[**Xiyu Zhang***](https://github.com/xyzhang77) · [**Chong Bao***](https://chobao.github.io) · [**Yipeng Chen**](https://openreview.net/profile?id=~YiPeng_Chen2) · [**Hongjia Zhai**](https://zhaihongjia.github.io/) · [**Yitong Dong**](https://openreview.net/profile?id=~Yitong_Dong1)
<br>
[**Hujun Bao**](http://www.cad.zju.edu.cn/home/bao) · [**Zhaopeng Cui**](https://zhpcui.github.io/) · [**Guofeng Zhang**](http://www.cad.zju.edu.cn/home/gfzhang/)<sup>&dagger;</sup> 
<br>

[![arXiv](https://img.shields.io/badge/arXiv-AtlasGS-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2510.25129)
[![Safari](https://img.shields.io/badge/Project_Page-AtlasGS-green?logo=safari&logoColor=fff)](https://zju3dv.github.io/AtlasGS)
[![Google Drive](https://img.shields.io/badge/Datasets-4285F4?logo=googledrive&logoColor=fff)](https://drive.google.com/drive/folders/1yRcA9sHlbqaH0pUQXhhjschCcnKdFXyo?usp=drive_link)

</div>

## News

- **[2025-12-31]**: Release the initial version of codes, datasets, checkpoints.

## TODO

- [x] Release codes.
- [x] Release datasets.
- [x] Realese checkpoints.
- [ ] Add viewer.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Mesh Extraction](#mesh-extraction)
5. [Evaluation](#evaluation)
6. [Pretrained Models](#pretrained-models)

## Installation

### Prerequisites
- CUDA 11.7 or higher
- Python 3.9
- PyTorch 1.13.1

### Step-by-Step Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/xyzhang77/AtlasGS.git
cd AtlasGS

# Create conda environment
conda create -n atlasgs python=3.9 -y
conda activate atlasgs

# Install PyTorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install -r requirements.txt

# Install custom rasterization module
pip install submodules/diff-surfel-rasterization --no-build-isolation

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --no-build-isolation
```


## Dataset Preparation

For detailed dataset preprocessing instructions, please refer to **[DATASET_PREPROCESSING.md](./DATASET_PREPROCESSING.md)**.

### Quick Start

**Option 1: Download Preprocessed Datasets**

We provide preprocessed datasets for immediate use:
- **[ScanNet](https://drive.google.com/file/d/14COBnkAVHIMlZawDNRX0rbEbu66vRpp1/view?usp=drive_link)**: scene0050_00, scene0084_00, scene0580_00, scene0616_00
- **[ScanNet++](https://drive.google.com/file/d/1VjF16BKDBXHOHaF0k3puymmoYcICBo4W/view?usp=drive_link)**: 8b5caf3398, b20a261fdf, f34d532901, f6659a3107
- **[Replica](https://drive.google.com/file/d/1Jv2YnVsR0rJXREk8ohGUjvsCRdsCCdrk/view?usp=drive_link)**: office0-3, room0-2

**Option 2: Process Custom Datasets**

For processing custom datasets (ScanNet, Replica, ScanNet++, COLMAP) and generating geometry priors (depth, normals, semantics), see the complete guide in **[DATASET_PREPROCESSING.md](./DATASET_PREPROCESSING.md)**.

### Basic Dataset Structure

After processing, each dataset should have the following structure:

```
data/
├── scene_name/
│   ├── images/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── sparse/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   ├── depths/ (optional)
│   ├── normals/ (optional)
│   └── semantics/ (optional)
```

### Training Examples

#### ScanNet Training
```bash
# Train on ScanNet scene
python train.py \
    -s data/scannet/scene0050_00 \
    -m output/scene0050_00 \
```

#### Replica Training
```bash
# Train on Replica scene
python train.py \
    -s data/replica/office0 \
    -m output/office0 \
```
## Mesh Extraction

```bash
python render.py -m <model_path> --skip_train --skip_test
```

### Rendering Images

```bash
# Render training views
python render.py -m output/scene0050_00

# Render test views only
python render.py -m output/scene0050_00 --skip_train

# Render interpolated camera trajectory
python render.py -m output/scene0050_00 --render_path
```

## Evaluation

Download ground truth meshes from [google drive](https://drive.google.com/file/d/1fI3WtaRMG1Hti4f1D-TGm-btI2Gk9PRN/view?usp=drive_link).

### ScanNet Evaluation

```bash
python -m evaluation.eval_scannet \
    --scene scene0050_00 \
    --mesh_path output/scene0050_00/train/ours_40000/fuse_post.ply
```

### ScanNet++ Evaluation

```bash
python -m evaluation.eval_scannetpp \
    --scene <scene_name> \
    --mesh_path <path_to_extracted_mesh>
```

### Replica Evaluation

```bash
python -m evaluation.eval_replica \
    --scene office0 \
    --mesh_path output/office0/train/ours_25000/fuse_post.ply
```

### Evaluation Metrics

The evaluation script computes the following metrics:

- **Accuracy (Acc)**: Mean distance from predicted to ground truth
- **Completeness (Comp)**: Mean distance from ground truth to predicted
- **Precision**: Percentage of points within threshold
- **Recall**: Percentage of ground truth points within threshold
- **F-score**: Harmonic mean of precision and recall

The evaluation also generates:
- Colored point clouds showing precision/recall errors
- Precision-Recall curves
- Detailed metrics saved as JSON

### Batch Evaluation

For multiple scenes, you can use the provided scripts:

```bash
# Evaluate all ScanNet scenes
python scripts/train_eval_scannet.py

# Evaluate all Replica scenes
python scripts/train_eval_replica.py

# Evaluate all ScanNet++ scenes
python scripts/train_eval_scannetpp.py
```

## Pretrained Models

We provide pretrained models for quick testing and evaluation without training from scratch.

### Available Models

- **ScanNet**: scene0050_00, scene0084_00, scene0580_00, scene0616_00
- **ScanNet++**: 8b5caf3398, b20a261fdf, f34d532901, f6659a3107  
- **Replica**: office0-3, room0-2

### Download and Usage

```bash
# Download pretrained models (link will be provided)
wget https://drive.google.com/your_pretrained_models_link

# Use for rendering
python render.py -m path/to/pretrained/model

# Use for mesh extraction
python render.py -m path/to/pretrained/model --skip_train --skip_test
```

## Acknowledgments

We acknowledge the following inspiring prior work:

- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)
- [Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering](https://city-super.github.io/scaffold-gs/)
- [Neural 3D Scene Reconstruction with the Manhattan-world Assumption](https://github.com/zju3dv/manhattan_sdf)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
zhang2025atlasgs,
title={Atlas{GS}: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians},
author={Xiyu Zhang and Chong Bao and YiPeng Chen and Hongjia Zhai and Yitong Dong and Hujun Bao and Zhaopeng Cui and Guofeng Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
```



---

For any questions or issues, please open an issue on the GitHub repository or contact the authors with the [email](xyzhang77@zju.edu.cn).
