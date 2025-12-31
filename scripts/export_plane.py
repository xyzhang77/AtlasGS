import numpy as np
import argparse
import open3d as o3d
import os
from utils.plane_utils import create_plane

def get_parser():
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--model_path', type=str, required=True, help='path to the plane file')
    paraser.add_argument('--iterations', type=int, default=-1, help='number of iterations')
    return paraser.parse_args()

def main():
    args = get_parser()
    iterations = max(int(i.split("_")[1]) for i in os.listdir(os.path.join(args.model_path, "point_cloud")))

    if args.iterations != -1:
        iterations = args.iterations
    
    plane_path = os.path.join(args.model_path, "point_cloud", f"iteration_{iterations}", "structure.txt")
    print(plane_path)
    output_path = os.path.join(args.model_path, "point_cloud", f"iteration_{iterations}", "plane.ply")

    plane_params = np.loadtxt(plane_path)
    # plane_params = [-0.0462,  0.0197, -0.9987, -0.2726, -2.9044]

    normal = np.array(plane_params[:3])
    normal = normal / np.linalg.norm(normal)
    floor = plane_params[3]
    ceiling = plane_params[4]

    plane_mesh_floor = create_plane(normal, -floor)
    o3d.io.write_triangle_mesh(output_path, plane_mesh_floor)

    plane_mesh_ceiling = create_plane(normal, -ceiling)
    o3d.io.write_triangle_mesh(output_path.replace('.ply', '_ceiling.ply'), plane_mesh_ceiling)
    return

if __name__ == "__main__":
    main()