import argparse
import os
import numpy as np
import subprocess

from PIL import Image
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, help='Input directory')
    parser.add_argument('--scenes', type=str, nargs='+', default=[],help='List of scenes to process')
    return parser.parse_args()

def initialize_cmd(gpu_id, require_envname, BLOCKING=False):
    env_name = os.getenv('CONDA_DEFAULT_ENV')
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} "
    if BLOCKING:
        cmd = " CUDA_LAUNCH_BLOCKING=1 " + cmd
    if env_name != require_envname:
        cmd += f"conda run -n {require_envname} --live-stream "
    return cmd

def exec_cmd(code_dir, cmd, run_cmd=True):
    if not run_cmd:
        return
    cmd = f"cd {code_dir} && {cmd}"
    subprocess.run(cmd, shell=True, check=True)

def run_stablenormal(gpu_id, image_paths, output_dir):
    import torch
    torch.set_default_device(f'cuda:{gpu_id}')
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    os.makedirs(output_dir, exist_ok=True)

    for path in tqdm(image_paths):
        input_image = Image.open(path)
        normal_image = predictor(input_image)
        out_path = os.path.join(output_dir, os.path.basename(input_image))[:-4] + ".png"
        normal_image.save(out_path)

def run_depthanythingv2_depth(gpu_id, image_folder, output_dir, metric=True):
    code_dir = "/mnt/nas_10/group/zhangxiyu/Archive-codes/Depth-Anything-V2"
    ckpt_path = "/mnt/nas_10/group/zhangxiyu/Archive/DepthAnythingv2/metric/depth_anything_v2_metric_hypersim_vitl.pth"
    assert os.path.exists(code_dir), "Please set correct code_dir for DepthAnythingV2"
    assert os.path.exists(ckpt_path), "Please set correct ckpt_path for DepthAnythingV2"

    if metric:
        code_dir += "/metric_depth"
    exe_name = "run.py"
    cmd = initialize_cmd(gpu_id)
    cmd +=(
           f"python {exe_name} "
           f"--encoder vitl " 
           f"--img-path {image_folder} "
           f"--outdir {output_dir} "  
           f"--pred-only "
           "--save-numpy "
           "--max-depth 20 "
           f"--load-from {ckpt_path} "
    )
    exec_cmd(code_dir, cmd)
    return

def process_single_scene(scene):
    images_folder = os.path.join(scene, "images")
    out_folder = scene
    image_names = os.listdir(images_folder)
    image_paths = [os.path.join(images_folder, image_name) for image_name in image_names]
    run_depthanythingv2_depth(0, image_paths, os.path.join(out_folder, "depths"))
    run_stablenormal(0, image_paths, os.path.join(out_folder, "normals"))

def main():
    args = get_parser()
    if len(args.scenes) == 0:
        args.scenes = os.listdir(args.scene_dir)

    for scene in args.scenes:
        process_single_scene(os.path.join(args.scene_dir, scene))
    return

if __name__ == "__main__":
    main()
