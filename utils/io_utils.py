from plyfile import PlyData, PlyElement
import numpy as np
import torch
import os, json
from PIL import Image
from .general_utils import build_rotation, LABEL_MAP


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_gs_ply(path, max_sh_degree = 3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rotation_matrices = build_rotation(torch.from_numpy(rots)).cpu().numpy()
    normals = rotation_matrices[..., :3, 2]
    return xyz, opacities, normals

WALL_PART = ["wall", "door", "window"]
FLOOR_PART = ["floor", "rug"]
CEILING_PART = ["ceiling"]

def load_semantic_seem(path, name):
    semantic = Image.open(os.path.join(path, name+".png"))
    semantic = np.array(semantic)
    with open(os.path.join(path, name+".json")) as json_file:
        json_data = json.load(json_file)
    
    ret_sem = np.zeros_like(semantic)

    for data in json_data:
        labeled = False
        mask_id = data['id']
        mask_label = data['category'].split("_")[0]
        # if mask_label in WALL_PART:
        #     ret_sem[semantic == mask_id] = LABEL_MAP["wall"]
        # elif mask_label in FLOOR_PART:
        #     ret_sem[semantic == mask_id] = LABEL_MAP["floor"]
        # elif mask_label in CEILING_PART:
        #     ret_sem[semantic == mask_id] = LABEL_MAP["ceiling"]
        # else:
        #     ret_sem[semantic == mask_id] = LABEL_MAP["others"]

        for wall_part in WALL_PART:
            if wall_part in mask_label:
                ret_sem[semantic == mask_id] = LABEL_MAP["wall"]
                labeled = True
                break

        for floor_part in FLOOR_PART:
            if floor_part in mask_label:
                ret_sem[semantic == mask_id] = LABEL_MAP["floor"]
                labeled = True
                break
            
        for ceiling_part in CEILING_PART:
            if ceiling_part in mask_label:
                ret_sem[semantic == mask_id] = LABEL_MAP["ceiling"]
                labeled = True
                break
        
        if not labeled:
            ret_sem[semantic == mask_id] = LABEL_MAP["others"]

    return ret_sem

def load_semantic_mask2former(path, name):
    data = np.load(os.path.join(path, name+".npz"))
    semantic = data["probabilities"]
    confidence = data["confidence"]
    return np.concatenate([semantic, confidence[..., None]], axis=-1)