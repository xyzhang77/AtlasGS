import numpy as np
import os, glob
import cv2
from PIL import Image
from tqdm import tqdm

colormap = np.array([
    [230, 57, 70],   # Deep Orange
    [42, 157, 143],  # Soft Green
    [69, 123, 157],  # Teal Blue
    [244, 162, 97]   # Warm Yellow
], dtype=np.uint8)


def main():
    path = "/data/zhangxiyu/projects/GSNroom/Results/replica_full-mask/sfm/room0/iter40000_normal_depth_sem0.5_struct0.1_w2D20000_dist100.0_jbwtv0.1_normal0.1_d0.25_densify_start1500_densify_end20000_sh0_detached_conf/train/ours_40000/vis"
    out_dir = "outputs/semantics"
    os.makedirs(out_dir, exist_ok=True)
    image_list = sorted(glob.glob(os.path.join(path, "semantic_*.png"), recursive=True))
    # print(image_list)
    for image_path in tqdm(image_list):
        sem_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # labels, label_ids = np.unique(sem_image, return_inverse=True)
        label_ids = np.zeros(sem_image.shape, dtype=int)
        label_ids[sem_image == 255] = 3
        label_ids[sem_image == 0] = 0
        label_ids[sem_image == 85] = 1
        label_ids[sem_image == 170] = 2


        # wall = label_ids == 0
        # floor = label_ids == 1
        # ceiling = label_ids == 2
        # others = label_ids == 3

        color = colormap[label_ids].reshape(*sem_image.shape, 3)#.transpose(1, 2, 0)
        Image.fromarray(color).save(os.path.join(out_dir, os.path.basename(image_path)))
        # quit()
        # labels = np.unique(sem_image.reshape(-1, 3), axis=0)
        # pri
    return

if __name__ == "__main__":
    main()