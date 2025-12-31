import os
import sys
import argparse
import PIL.Image
import numpy as np
import glob
from torchvision.transforms.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop
import cv2
from utils.colmap_utils import Camera, Image, rotmat2qvec, write_model
import PIL
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str, default="/mnt/nas_10/datasets/ScanNet_clean/scans", help='Input directory')
    parser.add_argument("--scenes", nargs="+", type=str, default=[], help="List of scenes to process")
    parser.add_argument('--outdir', type=str, default="/mnt/nas_10/group/zhangxiyu/datasets/scannet/scans", help='Output directory')
    parser.add_argument('--test', action="store_true", help='Output directory')

    return parser.parse_args()

transform = Compose([
    ToTensor(),
    CenterCrop([927, 1236]),
    Resize((480, 640)),
    ToPILImage()
])

def process_image(image, preprocess_image_fn = None):
    image = np.asarray(image)
    if preprocess_image_fn is not None:
        image = preprocess_image_fn(image)
    
    image = PIL.Image.fromarray(image)
    return transform(image)

def process_single_scene(scene_folder, outdir, test = False):
    os.makedirs(outdir, exist_ok=True)

    intrinsic = np.loadtxt(os.path.join(scene_folder, "intrinsic", "intrinsic_color.txt"))
    poses = []
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(scene_folder, "pose", '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    color_paths = sorted(glob.glob(os.path.join(scene_folder, "color", '*.jpg')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths:
        c2w = np.loadtxt(pose_path)
        poses.append(c2w)
    poses = np.array(poses)
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)

    H_ini, W_ini = 968, 1296
    H_crop, W_crop = 927, 1236
    H, W = 480, 640
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    new_cx, new_cy = W_ini / 2, H_ini / 2

    delta_cx = new_cx - cx
    delta_cy = new_cy - cy
    print(fx, fy, cx, cy, fx * W / W_crop, fy * H / H_crop)

    # center crop to (1236, 927)
    # resize to (480, 640)
    fx = fx * W / W_crop
    fy = fy * H / H_crop
    cx = (new_cx - (W_ini - W_crop) / 2) * W / W_crop
    cy = (new_cy - (H_ini - H_crop) / 2) * H / H_crop
    # Create the translation matrix
    print(fx, fy, cx, cy)
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

    for idx, (valid, pose, image_path) in enumerate(zip(valid_poses, poses, color_paths)):
        if idx % 10 != 0: 
            if valid:
                test_set.append(idx)
            continue
        if not valid : continue
        print(idx, valid)

        if test:
            continue
        
        target_image = os.path.join(outdir, "images", "%06d.png"%(out_index))
        out_index += 1
        extrinsic = np.linalg.inv(pose)
        qvec = rotmat2qvec(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3]
        image = Image(id=out_index, qvec=qvec, tvec=tvec, camera_id=1, name=os.path.basename(target_image), xys=[], point3D_ids=[])
        images[out_index] = image    
    

    if test:

        selected_view = np.random.choice(test_set, 50, replace=True)
        for idx, selected_image_id in enumerate(selected_view):
            pose = poses[selected_image_id]
            source_path = color_paths[selected_image_id]

            target_image = os.path.join(outdir, "images", "%06d.png"%(idx))
            os.makedirs(os.path.dirname(target_image), exist_ok=True)
            img = PIL.Image.open(source_path)

            img_tensor = process_image(img, preprocess_image_fn)
            img_tensor.save(target_image)
            extrinsic = np.linalg.inv(pose)
            qvec = rotmat2qvec(extrinsic[:3, :3])
            tvec = extrinsic[:3, 3]

            images[idx + 1] = Image(
                id = idx + 1,
                qvec = rotmat2qvec(np.linalg.inv(pose)[:3, :3]),
                tvec = tvec,
                camera_id = 1, 
                name = "%06d.png"%(idx),
                xys= [],
                point3D_ids=[]
            )
            
        os.makedirs(os.path.join(outdir, "sparse"), exist_ok=True)
        write_model(cameras, images, points, os.path.join(outdir, "sparse"), ext=".txt")
        write_model(cameras, images, points, os.path.join(outdir, "sparse"), ext=".bin")

    else:
        os.makedirs(os.path.join(outdir, "mannual_sparse"), exist_ok=True)
        write_model(cameras, images, points, os.path.join(outdir, "mannual_sparse"), ext=".txt")
        write_model(cameras, images, points, os.path.join(outdir, "mannual_sparse"), ext=".bin")

    return

def main():
    args = get_parser()
    if len(args.scenes) == 0:
        scenes = ["scene0050_00", "scene0084_00", "scene0580_00", "scene0616_00"]
    else:
        scenes = args.scenes
    for scene in scenes:
        process_single_scene(os.path.join(args.scan_dir, scene), os.path.join(args.outdir, scene), args.test)

    return


if __name__ == "__main__":
    main()