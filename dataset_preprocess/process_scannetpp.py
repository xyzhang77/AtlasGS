import os
import sys
import argparse
import numpy as np
import glob
from torchvision.transforms.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop
import cv2
from utils.colmap_utils import Camera, Image, rotmat2qvec, write_model, read_model
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str, default="/mnt/nas_10/datasets/scannetpp/data", help='Input directory')
    parser.add_argument("--scenes", nargs="+", type=str, default=["8b5caf3398", "b20a261fdf", "f6659a3107", "f34d532901"], help="List of scenes to process")
    parser.add_argument('--outdir', type=str, default="/mnt/nas_10/group/zhangxiyu/datasets/scannetpp", help='Output directory')
    parser.add_argument("--test", action="store_true", help="process for novel view synthesis")

    return parser.parse_args()

transform = Compose([
    ToTensor(),
    Resize((480, 640)),
    ToPILImage()
])

def process_image(image, preprocess_image_fn = None):
    image = np.asarray(image)
    if preprocess_image_fn is not None:
        image = preprocess_image_fn(image)
    
    image = PIL.Image.fromarray(image)
    return transform(image)

def process_single_scene(scene_folder, outdir, test=False):
    os.makedirs(outdir, exist_ok=True)
    cameras, images, _ = read_model(os.path.join(scene_folder, "colmap"), ext=".txt")
    images = dict(sorted(images.items(), key = lambda x : x[1].name))
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    width = cam.width
    height = cam.height
    assert width == 1920 and height == 1440, "Invalid image size" 
    fx, fy, cx, cy = cam.params[:4]

    desired_cx, desired_cy = width / 2, height / 2
    dx = desired_cx - cx
    dy = desired_cy - cy
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    transform_func = lambda x: cv2.warpAffine(x, translation_matrix, (width, height))
    out_index = 0
    fx = fx * 640 / (width - 80)
    fy = fy * 480 / (height - 60)
    cx = 320
    cy = 240

    cameras = {1: Camera(id=1, model="PINHOLE", width=640, height=480, params=[fx, fy, cx, cy])}
    # if test:
    #     out_image_folder = os.path.join(outdir, "test_images")
    # else:
    out_image_folder = os.path.join(outdir, "images")
    
    images_new = {}
    os.makedirs(out_image_folder, exist_ok=True)
    total_images = len(images)
    train_set = []
    test_set = []

    for idx, (image_id, image) in enumerate(tqdm(images.items())):
        if idx % 2 != 0:
            test_set.append(image_id)
            continue
        else:
            train_set.append(image_id)
        
        if test:
            continue
        # import pdbp; pdbp.set_trace()
        target_image = os.path.join(out_image_folder, "%06d.png"%(out_index))
        source_path = os.path.join(scene_folder, "rgb", image.name.split('/')[-1])
        rgb = cv2.imread(source_path)
        rgb = transform_func(rgb)
        rgb = rgb[40:-40, 30:-30]
        rgb = cv2.resize(rgb, (640, 480))
        cv2.imwrite(target_image, rgb)

        new_image = Image(
            id = out_index + 1,
            qvec = image.qvec,
            tvec = image.tvec,
            camera_id = 1,
            name = "%06d.png"%(out_index),
            xys = image.xys,
            point3D_ids = image.point3D_ids
        )
        
        out_index += 1
        images_new[out_index] = new_image    

    if test:
        selected_view = np.random.choice(test_set, 50, replace=True)
        for idx, selected_image_id in enumerate(selected_view):
            target_image = os.path.join(out_image_folder, "%06d.png"%(idx))
            source_path = os.path.join(scene_folder, "rgb", images[selected_image_id].name)
            rgb = cv2.imread(source_path)
            rgb = transform_func(rgb)
            rgb = rgb[40:-40, 30:-30]
            rgb = cv2.resize(rgb, (640, 480))
            cv2.imwrite(target_image, rgb)

            images_new[idx + 1] = Image(
                id = idx + 1,
                qvec = images[selected_image_id].qvec,
                tvec = images[selected_image_id].tvec,
                camera_id = 1,
                name = "%06d.png"%(idx),
                xys = images[selected_image_id].xys,
                point3D_ids = images[selected_image_id].point3D_ids
            )
            
        
    points = {}
    if test:
        out_sparse_dir = os.path.join(outdir, "sparse")
    else:
        out_sparse_dir = os.path.join(outdir, "mannual_sparse")
    os.makedirs(out_sparse_dir, exist_ok=True)
    write_model(cameras, images_new, points, out_sparse_dir, ext=".txt")
    write_model(cameras, images_new, points, out_sparse_dir, ext=".bin")

    return

def main():
    args = get_parser()
    scenes = args.scenes
    for scene in scenes:
        process_single_scene(os.path.join(args.scan_dir, scene, "iphone"), os.path.join(args.outdir, scene), args.test)

    return

if __name__ == "__main__":
    main()