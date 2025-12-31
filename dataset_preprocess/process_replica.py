import os
import sys
import argparse
import numpy as np
import glob
from torchvision.transforms.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop
import cv2
import json
from utils.colmap_utils import Camera, Image, rotmat2qvec, write_model
import PIL
import matplotlib.pyplot as plt
import shutil


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str, default="/mnt/nas_10/group/zhangxiyu/datasets/Replica", help='Input directory')
    parser.add_argument("--scenes", nargs="+", type=str, default=["office0", "office1", "office2", "office3", "room0", "room1", "room2"], help="List of scenes to process")
    parser.add_argument('--outdir', type=str, default="/mnt/nas_10/group/zhangxiyu/datasets/replica-gs", help='Output directory')
    parser.add_argument('--test', action="store_true", help='Output directory')

    return parser.parse_args()

transform = Compose([
    ToTensor(),
    CenterCrop([672, 896]),
    Resize((480, 640)),
    ToPILImage()
])

def process_image(image, preprocess_image_fn = None):
    image = np.asarray(image)
    if preprocess_image_fn is not None:
        image = preprocess_image_fn(image)
    
    image = PIL.Image.fromarray(image)
    return transform(image)

def load_replica_pose(traj_path):
    traj = np.loadtxt(traj_path)
    traj = traj.reshape(-1, 4, 4)
    return traj

def process_single_scene(scene_folder, outdir, intrinsic = None, test = False):
    os.makedirs(outdir, exist_ok=True)
    assert intrinsic is not None, "Intrinsic matrix is not provided"
    poses = load_replica_pose(os.path.join(scene_folder, "traj.txt"))
    color_paths = sorted(glob.glob(os.path.join(scene_folder, "results", 'frame*.jpg')))
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)

    H_ini, W_ini = 680, 1200
    H_crop, W_crop = 672, 896
    H, W = 480, 640
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    new_cx, new_cy = W_ini / 2, H_ini / 2

    delta_cx = new_cx - cx
    delta_cy = new_cy - cy

    # center crop to (1236, 927)
    # resize to (480, 640)
    fx = fx * W / W_crop
    fy = fy * H / H_crop
    cx = (new_cx - (W_ini - W_crop) / 2) * W / W_crop
    cy = (new_cy - (H_ini - H_crop) / 2)* H / H_crop
    # Create the translation matrix
    translation_matrix = np.array([
        [1, 0, delta_cx],
        [0, 1, delta_cy]
    ], dtype=np.float32)

    # Apply the transformation
    preprocess_image_fn = lambda image: cv2.warpAffine(image, translation_matrix, (W_ini, H_ini))

    out_index = 0
    cameras = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=[fx, fy, cx, cy])}
    images = {}
    points = {}
    train_set = []
    test_set = []

    print(valid_poses.shape, poses.shape, len(color_paths))
    for idx, (valid, pose, image_path) in enumerate(zip(valid_poses, poses, color_paths)):
        if test:
            if idx % 10 != 0 and valid: 
                test_set.append(idx)
            elif idx % 10 == 0 and valid:
                train_set.append(idx)
            
            continue
        else:
            if idx % 10 != 0: continue
            if not valid : continue

        train_set.append(idx)
        
        target_image = os.path.join(outdir, "images", "%06d.png"%(out_index))
        os.makedirs(os.path.dirname(target_image), exist_ok=True)
        print(target_image)
        img = PIL.Image.open(image_path)
        img_tensor = process_image(img, preprocess_image_fn)
        img_tensor.save(target_image)
        out_index += 1
        extrinsic = np.linalg.inv(pose)
        qvec = rotmat2qvec(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3]
        image = Image(id=out_index, qvec=qvec, tvec=tvec, camera_id=1, name=os.path.basename(target_image), xys=[], point3D_ids=[])
        images[out_index] = image    
    
    if test:
        images_test = {}
        selected_view = np.random.choice(test_set, 50, replace=True)
        out_index = 0
        for idx, selected_image_id in enumerate(selected_view):
            pose = poses[selected_image_id]
            rgb_path = color_paths[selected_image_id]

            target_image = os.path.join(outdir, "images", "%06d.png"%(out_index))
            os.makedirs(os.path.dirname(target_image), exist_ok=True)
            print(target_image)
            img = PIL.Image.open(rgb_path)
            img_tensor = process_image(img, preprocess_image_fn)
            img_tensor.save(target_image)
            out_index += 1
            extrinsic = np.linalg.inv(pose)
            qvec = rotmat2qvec(extrinsic[:3, :3])
            tvec = extrinsic[:3, 3]
            image = Image(id=out_index, qvec=qvec, 
                          tvec=tvec, camera_id=1, 
                          name=os.path.basename(target_image), 
                          xys=[], point3D_ids=[])

            images[out_index] = image    

        os.makedirs(os.path.join(outdir, "sparse"), exist_ok=True)
        write_model(cameras, images, points, os.path.join(outdir, "sparse"), ext=".txt")
        write_model(cameras, images, points, os.path.join(outdir, "sparse"), ext=".bin")
        scene = os.path.basename(scene_folder)
        shutil.copy(f"/mnt/nas_10/group/zhangxiyu/datasets/replica-gs/{scene}/sparse/points3D.bin", os.path.join(outdir, "sparse"))
        shutil.copy(f"/mnt/nas_10/group/zhangxiyu/datasets/replica-gs/{scene}/sparse/points3D.txt", os.path.join(outdir, "sparse"))
    else:
        os.makedirs(os.path.join(outdir, "mannual_sparse"), exist_ok=True)
        write_model(cameras, images, points, os.path.join(outdir, "mannual_sparse"), ext=".txt")
        write_model(cameras, images, points, os.path.join(outdir, "mannual_sparse"), ext=".bin")

    return

def main():
    args = get_parser()
    with open(os.path.join(args.scan_dir, "cam_params.json"), "r") as f:
        cam_params = json.load(f)["camera"]
    cx = cam_params["cx"]
    cy = cam_params["cy"]
    fx = cam_params["fx"]
    fy = cam_params["fy"]
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    for scene in args.scenes:
        process_single_scene(os.path.join(args.scan_dir, scene), os.path.join(args.outdir, scene), intrinsic, test = args.test)

    return


if __name__ == "__main__":
    main()