import numpy as np
import argparse
import os
import PIL
import cv2
import torch
from utils.colmap_utils import read_model, qvec2rotmat
from utils.io_utils import load_semantic_mask2former
from utils.plane_utils import fit_plane_ransac, create_plane
from tqdm import tqdm
import open3d as o3d

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--scenes", type=str, nargs= "+", default = None)

    return parser.parse_args()

def process_single_scene(path):

    colmap_dir = os.path.join(path, "sparse")
    semantic_dir = os.path.join(path, "panoptic")

    assert os.path.exists(colmap_dir), "Colmap directory does not exist"
    assert os.path.exists(semantic_dir), "Semantic directory does not exist"
    
    cameras, images, points3D = read_model(colmap_dir)
    points = np.zeros((max(points3D.keys()) + 1, 3))
    print(points.shape)

    for point_id, point in points3D.items():
        points[point_id] = point.xyz
    
    semantics3D = np.zeros((points.shape[0], 4), dtype = float)
    count = np.zeros(points.shape[0])
    cam_loc = []
    for image_id, image in tqdm(images.items()):
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        pose = np.linalg.inv(pose)
        cam_loc.append(pose[:3, 3])

        name = image.name.split('.')[0]
        valid = image.point3D_ids > 0
        if valid.sum() == 0:
            continue
        xys = image.xys[valid].astype(np.float32)
        semantic = load_semantic_mask2former(semantic_dir, name)
        semantic = semantic[..., :4]
        semantic = cv2.remap(semantic, xys[:, :1], xys[:, 1:], cv2.INTER_LINEAR)
        ids = image.point3D_ids[valid]
        semantics3D[ids] += semantic.squeeze()
        count[ids] += 1

    cam_loc = np.array(cam_loc)
    scene_center = (cam_loc.max(0) + cam_loc.min(0)) / 2

    valid_points = count > 0
    points = points[valid_points]
    semantics3D = semantics3D[valid_points]
    semantics3D = semantics3D.argmax(axis=1)

    is_floor = semantics3D == 1
    is_wall = semantics3D == 0
    is_ceiling = semantics3D == 2
    
    points = torch.from_numpy(points).float().cuda()
    plane, _ = fit_plane_ransac(points[is_floor])

    floor_plane = plane[0] / torch.norm(plane[0, :3])
    scene_center = torch.from_numpy(scene_center).cuda()
    floor_plane = torch.sign((floor_plane[:3] * scene_center).sum(-1)) * floor_plane
    structure_normal = floor_plane[:3]
    floor_distance = -floor_plane[3]

    if is_ceiling.sum() == 0:
        ceiling_distance = torch.tensor(0.0, dtype=torch.float, device="cuda")
    else:
        ceiling_distance = (points[is_ceiling] * structure_normal).sum(-1).mean()

    structure_normal = structure_normal.cpu().numpy()
    floor_distance = floor_distance.cpu().numpy()
    ceiling_distance = ceiling_distance.cpu().numpy()
    np.savetxt(os.path.join(path, "sparse", "structure_normal.txt"), 
               np.concatenate([structure_normal, [floor_distance, ceiling_distance]]))
    plane_mesh_floor = create_plane(structure_normal, -floor_distance, 3)
    o3d.io.write_triangle_mesh(os.path.join(path, "sparse", "floor.ply"), plane_mesh_floor)

    plane_mesh_ceiling = create_plane(structure_normal, -ceiling_distance, 3)
    o3d.io.write_triangle_mesh(os.path.join(path, "sparse", "ceiling.ply"), plane_mesh_ceiling)

def main():
    args = get_parser()
    scenes = os.listdir(args.path) if args.scenes is None else args.scenes
    for scene in scenes:
        process_single_scene(os.path.join(args.path, scene))
if __name__ == "__main__":
    main()
