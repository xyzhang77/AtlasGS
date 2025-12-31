#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch.nn.functional as F
import torch
from PIL import Image
from scene.cameras import Camera
import numpy as np
from tqdm import tqdm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.io_utils import load_semantic_seem, load_semantic_mask2former

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    if args.normals:
        normal_path = os.path.join(args.source_path, args.normals, os.path.basename(cam_info.image_path))
        
        if os.path.exists(normal_path[:-4]+ '.npy'):
            _normal = torch.tensor(np.load(normal_path[:-4]+ '.npy'))
            _normal = - (_normal * 2 - 1)
            resized_normal = F.interpolate(_normal.unsqueeze(0), size=resolution[::-1], mode='bicubic')
            _normal = resized_normal.squeeze(0)
        elif os.path.exists(normal_path[:-4]+ '.png'):
            _normal = Image.open(normal_path[:-4]+ '.png')
            resized_normal = PILtoTorch(_normal, resolution)
            resized_normal = resized_normal[:3]
            _normal = - (resized_normal * 2 - 1)
        else:
            _normal = None
        
        # normalize normal
        if _normal is not None:
            _normal = F.normalize(_normal, p=2, dim=0)
            _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
            _normal = _normal.permute(2, 0, 1)
    else:
        _normal = None

    if args.depths:
        depth_path = os.path.join(args.source_path, args.depths, os.path.basename(cam_info.image_path)[:-4]+ '.npy')
        depths = torch.from_numpy(np.load(depth_path)).unsqueeze(0)
        masks = (depths < 20) & (depths > 0)
    else:
        masks = None
        depths = None
    
    if args.semantics:
        semantic = load_semantic_mask2former(os.path.join(args.source_path, args.semantics), os.path.basename(cam_info.image_path)[:-4])
        semantic = torch.from_numpy(semantic).permute(2, 0, 1)
    else:
        semantic = None


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, normal = _normal, semantic = semantic,
                  image=gt_image, gt_alpha_mask=loaded_mask, mask = masks, depth = depths,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    if len(cam_infos) == 0:
        return []

    _ = loadCam(args, 0, cam_infos[0], resolution_scale)
    def load_single_cam(idx_info_tuple):
        idx, c = idx_info_tuple
        return idx, loadCam(args, idx, c, resolution_scale)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(load_single_cam, enumerate(cam_infos)), total=len(cam_infos)))

    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry