# adapted from https://github.com/zju3dv/manhattan_sdf
import os
os.environ["MKL_NUM_THREADS"] = "4" # noqa
os.environ["OMP_NUM_THREADS"] = "4" # noqa 

import numpy as np
import open3d as o3d
import trimesh
import torch
import glob
import pyrender
import os
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree
from cycler import cycler
from utils.colmap_utils import (
    read_cameras_text, 
    read_images_text,
    qvec2rotmat
)

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def write_color_distances(path, pcd, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def get_f1_score_histo2(threshold,
                        distance1,
                        distance2,
                        plot_stretch = 5,
                        ):
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]


def plot_graph(
    out_dir,
    dist_threshold,
    dist1, 
    dist2,
    plot_stretch = 5,
    show_figure=False,
):
    
    ( precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target ) = get_f1_score_histo2(dist_threshold, dist1, dist2)

    f = plt.figure()
    plt_size = [14, 7]
    pfontsize = "medium"

    ax = plt.subplot(111)
    label_str = "precision"
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "recall"
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)
    plt.rcParams["figure.figsize"] = plt_size
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))
    plt.title("Precision and Recall: " + "%02.2f f-score" %
              (fscore * 100))
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)

    plt.ylabel("# of points (%)", fontsize=15)
    plt.xlabel("Meters", fontsize=15)
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)
    plt.legend(loc=4)
    leg = plt.legend(loc="lower right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)
    png_name = out_dir + "/PR_@d_th_0_{0}.png".format(
        "%04d" % (dist_threshold * 10000))
    pdf_name = out_dir + "/PR_@d_th_0_{0}.pdf".format(
        "%04d" % (dist_threshold * 10000))

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02, save_path=None):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    if save_path:
        write_color_distances(save_path + "/precision.ply", pcd_pred, dist2, 3 * threshold)
        write_color_distances(save_path + "/recall.ply", pcd_trgt, dist1, 3 * threshold)

    plot_graph(
        save_path,
        threshold,
        dist2,
        dist1
    )

    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    with open(os.path.join(save_path, 'evaluation.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

# hard-coded image size
# H, W = 968, 1296

# load pose
def load_poses(scene_path):
    images = read_images_text(os.path.join(scene_path, "sparse", "images.txt"))
    poses = []
    
    for image_id, image in images.items():
        pose = np.eye(4)    
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        pose = np.linalg.inv(pose)
        poses.append(pose)
        
    poses = np.array(poses)
    return poses

def load_intrinsic(scene_path):
    cameras = read_cameras_text(os.path.join(scene_path, "sparse", "cameras.txt"))
    camera = cameras[1]
    width = camera.width
    height = camera.height

    fx, fy, cx, cy = camera.params[:4]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return width, height, intrinsic


class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def refuse(mesh, poses, K, height, width):
    renderer = Renderer(height, width)
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((height, width, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument('--data_dir', type=str, default="data/scannetpp")
    parser.add_argument('--gt_mesh', type=str, default=None)
    parser.add_argument('--result_folder', type=str, default=None)
    parser.add_argument("--norefuse", action="store_true")
    args = parser.parse_args()

    root_dir = "/mnt/nas_10/datasets/scannetpp/data"

    if args.result_folder is None:
        result_folder = os.path.join(os.path.dirname(args.mesh_path), f"evaluation_{args.scene}")
    else:
        result_folder = args.result_folder
    
    os.makedirs(result_folder, exist_ok=True)
    mesh = trimesh.load(args.mesh_path)
    if not args.norefuse:
        poses = load_poses(os.path.join(args.data_dir, args.scene))
        H, W, K = load_intrinsic(os.path.join(args.data_dir, args.scene))
        mesh = refuse(mesh, poses, K, H, W)

        # save mesh
        # out_mesh_path = args.mesh_path.replace(".ply", "_refused.ply")
        out_mesh_path = os.path.join(result_folder, "refused.ply")
        o3d.io.write_triangle_mesh(out_mesh_path, mesh)
        mesh = trimesh.load(out_mesh_path)
    
    if args.gt_mesh is not None:
        gt_mesh = args.gt_mesh
    else:
        gt_mesh = os.path.join("data", "gts", f"{args.scene}.ply")
    # gt_mesh = os.path.join()
    
    gt_mesh = trimesh.load(gt_mesh)

    metrics = evaluate(mesh, gt_mesh, save_path=result_folder)
    print(metrics)