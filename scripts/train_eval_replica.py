import os

scenes = ["office0", "office1", "office2", "office3", "room0", "room1", "room2"]
for scene in scenes:

    os.system(f"python train.py -s data/replica/{scene} -m output/{scene} --config configs/{scene}.yaml")
    os.system(f"python render.py -m output/{scene}/ --skip_train --skip_test")
    os.system(f"python -m evaluation.eval_replica --gt_mesh gts/replica/{scene}.ply --rec_mesh output/{scene}/train/ours_40000/fuse_post.ply")
