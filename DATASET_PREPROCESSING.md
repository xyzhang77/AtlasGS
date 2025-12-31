# Dataset Preprocessing Guide for AtlasGS

This guide provides comprehensive instructions for processing various datasets to work with AtlasGS, including basic processing and geometry prior generation.

## Table of Contents

1. [Download Preprocessed Datasets](#download-preprocessed-datasets)
2. [Basic Dataset Processing](#basic-dataset-processing)
3. [Geometry Prior Processing](#geometry-prior-processing)
4. [Dataset Structure](#dataset-structure)

## Download Preprocessed Datasets

We provide preprocessed datasets for immediate use:

- **[ScanNet](https://drive.google.com/file/d/14COBnkAVHIMlZawDNRX0rbEbu66vRpp1/view?usp=drive_link)**: scene0050_00, scene0084_00, scene0580_00, scene0616_00
- **[ScanNet++](https://drive.google.com/file/d/1VjF16BKDBXHOHaF0k3puymmoYcICBo4W/view?usp=drive_link)**: 8b5caf3398, b20a261fdf, f34d532901, f6659a3107
- **[Replica](https://drive.google.com/file/d/1Jv2YnVsR0rJXREk8ohGUjvsCRdsCCdrk/view?usp=drive_link)**: office0-3, room0-2

## Basic Dataset Processing

### ScanNet Dataset Processing

```bash
python dataset_preprocess/process_scannet.py \
    --scan_dir /path/to/scannet/scans \
    --scenes scene0050_00 scene0084_00 \
    --outdir /path/to/output/scannet
```

### Replica Dataset Processing

```bash
python dataset_preprocess/process_replica.py \
    --scan_dir /path/to/replica \
    --scenes office0 office1 office2 \
    --outdir /path/to/output/replica
```

### ScanNet++ Dataset Processing

```bash
python dataset_preprocess/process_scannetpp.py \
    --scan_dir /path/to/scannetpp \
    --outdir /path/to/output/scannetpp
```

### COLMAP Dataset Processing

For datasets with COLMAP reconstruction:

```bash
python dataset_preprocess/process_colmap.py \
    --scan_dir /path/to/dataset \
    --scenes scene1 scene2
```

## Geometry Prior Processing

AtlasGS requires geometry priors (depth, normals, and semantics) for improved reconstruction quality.

### Depth and Normal Priors

Use the `infer_priors.py` script to generate depth and normal maps from RGB images:

```bash
python dataset_preprocess/infer_priors.py \
    --scene_dir /path/to/processed/dataset \
    --scenes scene1 scene2
```

**Prerequisites for depth/normal processing:**
- **Depth-Anything-V2**: Install from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **StableNormal**: Install from [StableNormal](https://github.com/Stable-X/StableNormal)
- Download pretrained models:
  - Depth-Anything-V2 metric depth model: `depth_anything_v2_metric_hypersim_vitl.pth`

### Semantic Priors

For semantic processing, you need to use mask2former following [panoptic-lifting](https://github.com/nihalsid/panoptic-lifting) to generate semantic masks, then convert them to required format:

**Step 1: Generate semantic masks following panoptic-lifting**
```bash
# This step should be done using panoptic-lifting pipeline
# Output should be in .ptz format in the panoptic/ directory
```

**Step 2: Convert .ptz to .npz format**
```bash
python dataset_preprocess/process_semantic.py \
    --input /path/to/dataset
```

**Step 3: Visualize semantic colors (optional)**
```bash
python dataset_preprocess/process_semantic_to_color.py
```

**Semantic classes expected:**
- Class 0: Wall
- Class 1: Floor  
- Class 2: Ceiling
- Class 3: Others/Unlabeled

### Structure Extraction

Extract floor and ceiling planes from semantic data and COLMAP reconstruction:

```bash
python dataset_preprocess/extract_structure.py \
    --path /path/to/dataset \
    --scenes scene1 scene2
```

This script:
- Loads COLMAP sparse reconstruction
- Projects semantic labels to 3D points
- Fits RANSAC planes to floor and ceiling points
- Saves structure normal and distances
- Exports floor and ceiling mesh files

**Output files:**
- `sparse/structure_normal.txt`: Structure normal and distances
- `sparse/floor.ply`: Floor plane mesh
- `sparse/ceiling.ply`: Ceiling plane mesh

### Complete Prior Processing Pipeline

For a complete pipeline with all priors:

```bash
# 1. Basic dataset processing (choose based on dataset type)
python dataset_preprocess/process_scannet.py --scan_dir /path/to/scannet --scenes scene1

# 2. Generate geometry priors
python dataset_preprocess/infer_priors.py --scene_dir /path/to/processed/scene1 --scenes scene1

# 3. Process semantic data (if available)
python dataset_preprocess/process_semantic.py --input /path/to/processed/scene1

# 4. Extract structure information
python dataset_preprocess/extract_structure.py --path /path/to/processed --scenes scene1
```

## Dataset Structure

### Basic Structure

After basic processing, each dataset should have:

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
```

### Complete Structure with Priors

After complete processing with geometry priors, your dataset should have:

```
data/
├── scene_name/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   ├── sparse/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   ├── points3D.bin
│   │   ├── structure_normal.txt
│   │   ├── floor.ply
│   │   └── ceiling.ply
│   ├── depths/
│   │   ├── 000000.npy
│   │   └── ...
│   ├── normals/
│   │   ├── 000000.png
│   │   └── ...
│   ├── panoptic/
│   │   ├── 000000.npz
│   │   └── ...
│   └── semantics/ (optional, processed from panoptic)
```
