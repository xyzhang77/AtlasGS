import os

scenes = ["8b5caf3398", "b20a261fdf", "f34d532901", "f6659a3107"]
for scene in scenes:

    os.system(f"python train.py -s data/scannetpp/{scene} -m output/{scene}")
    os.system(f"python render.py -m output/{scene}/ --skip_train --skip_test")
    os.system(f"python -m evaluation.eval_scannetpp --scene {scene} --mesh_path output/{scene}/train/ours_40000/fuse_post.ply")
