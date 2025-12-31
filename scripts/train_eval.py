import os

scenes = ["scene0050_00", "scene0084_00", "scene0580_00", "scene0616_00"]
scenes = ["scene0616_00"]
for scene in scenes:

    os.system(f"python train.py -s data/scannet/{scene} -m output/{scene} --config configs/{scene}.yaml")
    os.system(f"python render.py -m output/{scene}/ --skip_train --skip_test")
    os.system(f"python -m evaluation.eval_scannet --scene {scene} --mesh_path output/{scene}/train/ours_40000/fuse_post.ply")
