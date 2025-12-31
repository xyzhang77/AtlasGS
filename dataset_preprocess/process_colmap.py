import argparse
import os
from utils.colmap_db_utils import COLMAPDatabase
from utils.colmap_utils import read_model, write_model, Image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str, default="/mnt/nas_10/datasets/ScanNet_clean/scans", help='Input directory')
    parser.add_argument("--scenes", nargs="+", type=str, default=[], help="List of scenes to process")

    return parser.parse_args()

def process_single_scene(scene_dir, outdir):
    workspace = os.path.join(outdir, "workspace")
    os.makedirs(workspace, exist_ok=True)

    os.system(f"colmap feature_extractor " \
               f"--database_path {workspace}/database.db " \
               f"--image_path {outdir}/images/ "\
               "--ImageReader.camera_model PINHOLE "\
               "--ImageReader.single_camera 1 ")

    os.system(f"colmap exhaustive_matcher  --database_path {workspace}/database.db")
    db = COLMAPDatabase.connect(f'{workspace}/database.db')


    cameras, ori_images, points = read_model(os.path.join(outdir, "mannual_sparse"), ext=".txt")

    images = list(db.execute('select * from images'))

    name_to_id = {image.name: image.id for image in ori_images.values()}
    new_images = {}
    for image in images:
        image_id = image[0]
        image_name = image[1]
        new_images[image_id] = Image(
            id=image_id, qvec=ori_images[name_to_id[image_name]].qvec, tvec=ori_images[name_to_id[image_name]].tvec, camera_id=ori_images[name_to_id[image_name]].camera_id,
            name=image_name, xys=ori_images[name_to_id[image_name]].xys, point3D_ids=ori_images[name_to_id[image_name]].point3D_ids
        )

    os.makedirs(os.path.join(workspace, "sparse"), exist_ok=True)
    write_model(cameras, new_images, points, os.path.join(workspace, "sparse"), ext=".txt")
    # quit()
    
    os.system(f" colmap point_triangulator "\
               f"--database_path {workspace}/database.db "\
               f"--image_path {outdir}/images "\
               f"--input_path {workspace}/sparse " \
               f"--output_path {workspace}/sparse/")

    os.system(f"colmap model_converter --input_path {workspace}/sparse --output_path {workspace}/sparse --output_type TXT")
    os.system(f"cp -r {workspace}/sparse {outdir}/sparse")
    return

def main():
    args = get_parser()
    scenes = args.scenes
    if len(scenes) == 0:
        scenes = os.listdir(args.scan_dir)
    
    for scene in scenes:
        process_single_scene(os.path.join(args.scan_dir, scene), os.path.join(args.scan_dir, scene))

    return

if __name__ == "__main__":
    main()