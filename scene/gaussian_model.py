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
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import open3d as o3d

from torch import nn
from einops import repeat
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid, build_rotation
from utils.io_utils import storePly
from utils.plane_utils import fit_plane_ransac, create_plane

class GaussianModel:

    def __init__(self, args):

        self.args = args
        
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._anchor_sem_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.structure = torch.empty(0)

        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.max_sh_degree = args.sh_degree
                
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        # Model configuration parameters (now from ModelParams)
        self.feat_dim = args.feat_dim
        self.view_dim = getattr(args, 'view_dim', 3)
        self.padding = args.padding
        self.semantic_dim = args.semantic_dim
        self.n_offsets = args.n_offsets
        self.voxel_size = args.voxel_size
        self.update_depth = args.update_depth
        self.update_init_factor = args.update_init_factor
        self.update_hierachy_factor = args.update_hierachy_factor
        self.alpha = args.alpha
        self.downsample_factor = args.downsample_factor
        
        # Other model parameters
        self.scale_dim = 2
        self.cached_gs = None
        self.view_embedding = 0

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_embedding, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, (4 + self.scale_dim)*self.n_offsets),
        ).cuda()
    
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_sem = nn.Sequential(
            nn.Linear(self.semantic_dim + self.view_embedding, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 4*self.n_offsets),
        ).cuda()

        self.structure = nn.Parameter(torch.tensor([0.0] * 5, dtype=torch.float, device="cuda")).requires_grad_(True)

        self.initialize_mlp()

    def initialize_mlp(self):
        nn.init.normal_(self.mlp_opacity[0].weight, 0, 0.001)
        nn.init.constant_(self.mlp_opacity[0].bias, 0)
        nn.init.normal_(self.mlp_opacity[2].weight, 0, 0.001)
        nn.init.constant_(self.mlp_opacity[2].bias, getattr(self, "bias_init", 0.1))

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        param_dict['sem_mlp'] = self.mlp_sem.state_dict()

        return (
            self._anchor,
            self._offset,
            self._scaling,
            self._rotation,
            self._anchor_feat,
            self._anchor_sem_feat,
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            self.structure,
            param_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._anchor,
        self._offset,
        self._scaling,
        self._rotation,
        self._anchor_feat,
        self._anchor_sem_feat,
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        self.structure,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        self.mlp_sem.load_state_dict(param_dict['sem_mlp'])

    @property
    def get_structure(self):
        normal = F.normalize(self.structure[:3], p=2, dim=0)

        return {
            "structure_normal": normal,
            "structure_distance": self.structure[3:],
        }

    @property
    def get_anchor(self):
        return self._anchor
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def get_anchor_semantic_feat(self):
        return self._anchor_sem_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    def get_semantic_mlp(self, semantic_feat):
        return self.mlp_sem(semantic_feat)

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_all_opacities(self):
        return self.mlp_opacity(self._anchor_feat)

    def voxelize_sample(self, data=None, voxel_size=0.001):
        if getattr(self, "downsample_factor", 1) > 1:
            print(f"Downsample factor: {self.downsample_factor}")
            voxel_size = voxel_size * self.downsample_factor
        data = torch.unique(torch.round(data/voxel_size), dim=0) * voxel_size 
        data += self.padding * voxel_size
        return data

    def create_from_pcd(self, pcd, spatial_lr_scale, *args):
        self.spatial_lr_scale = spatial_lr_scale
        # import pdbp; pdbp.set_trace()
        points = torch.tensor(pcd.points).float().cuda()
        if self.voxel_size <= 0:
            init_dist = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            torch.cuda.empty_cache()
                        
        fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size)
        
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        anchors_sem_feat = torch.zeros((fused_point_cloud.shape[0], self.semantic_dim)).float().cuda()
        
        print(f"N_offsets: {self.n_offsets}")
        print(f'Initial Voxel Number: {fused_point_cloud.shape[0]}')
        print(f'Voxel Size: {self.voxel_size}')

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3 + self.scale_dim)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._anchor_sem_feat = nn.Parameter(anchors_sem_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

        if self.args.use_colmap:
            structure = self.init_structure_from_colmap(
                os.path.join(self.args.source_path, "sparse/structure_normal.txt"),
            )
        else:
            structure = np.zero(5)

        self.structure = nn.Parameter(
            torch.tensor(structure, dtype=torch.float, device="cuda").requires_grad_(True)
        )

    def training_setup(self, training_args):
        
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._anchor_sem_feat], 'lr': training_args.feature_lr, "name": "anchor_sem_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.structure], 'lr': training_args.structure_lr_init, "name": "structure"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_sem.parameters(), 'lr': training_args.mlp_sem_lr_init, "name": "mlp_sem"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)

        self.structure_scheduler_args = get_expon_lr_func(lr_init=training_args.structure_lr_init,
                                                    lr_final=training_args.structure_lr_final,
                                                    lr_delay_mult=training_args.structure_lr_delay_mult,
                                                    start_step=training_args.loss_structure_begin,
                                                    max_steps = training_args.iterations - training_args.loss_structure_begin)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        self.mlp_sem_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_sem_lr_init,
                                                    lr_final=training_args.mlp_sem_lr_final,
                                                    lr_delay_mult=training_args.mlp_sem_lr_delay_mult,
                                                    max_steps=training_args.mlp_sem_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr_dict = {}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            
            if param_group["name"] == "structure":
                lr = self.structure_scheduler_args(iteration)
                param_group['lr'] = lr

            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            
            if param_group["name"] == "mlp_sem":
                lr = self.mlp_sem_scheduler_args(iteration)
                param_group['lr'] = lr

            lr_dict[param_group["name"]] = lr
        return lr_dict
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._anchor_sem_feat.shape[1]):
            l.append('f_anchor_sem_feat_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, iteration):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        anchor_sem_feat = self._anchor_sem_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, anchor_feat, anchor_sem_feat, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        xyz, colors, _, opacities, _, _, _, _ = self.generate_neural_gaussians()

        xyz = xyz.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        opacities = opacities.detach().cpu().numpy()

        alpha = np.minimum(opacities, 0.99) # (N, 1)
        vis_color = np.clip(colors * 255, 0, 255) # (N, 3)
        noempty_mask = alpha.squeeze() > 0.05
        vis_color = vis_color[noempty_mask]
        vis_xyz = xyz[noempty_mask]
        storePly(path.replace(".ply", "_vis.ply"), vis_xyz, vis_color)

        cmap = plt.get_cmap('hot')
        data_color = np.clip(cmap(opacities[..., 0])[:, :3] * 255, 0, 255)
        # print(data_color.shape, xyz.shape)
        storePly(path.replace(".ply", "_density.ply"), xyz, data_color)

        structure_np = self.structure.data.cpu().clone().numpy()
        np.savetxt(os.path.join(os.path.dirname(path), "structure.txt"), structure_np)

        structure_normal = structure_np[:3] / np.linalg.norm(structure_np[:3])
        plane_mesh_floor = create_plane(structure_normal, -structure_np[3])
        o3d.io.write_triangle_mesh(path.replace(".ply", "_floor.ply"), plane_mesh_floor)

        plane_mesh_ceiling = create_plane(structure_normal, -structure_np[4])
        o3d.io.write_triangle_mesh(path.replace(".ply", "_ceiling.ply"), plane_mesh_ceiling)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_sem_feat
        anchor_sem_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_sem_feat")]
        anchor_sem_feat_names = sorted(anchor_sem_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_sem_feats = np.zeros((anchor.shape[0], len(anchor_sem_feat_names)))
        for idx, attr_name in enumerate(anchor_sem_feat_names):
            anchor_sem_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
    
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_sem_feat = nn.Parameter(torch.tensor(anchor_sem_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._anchor_sem_feat = optimizable_tensors["anchor_sem_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
    
    
    def anchor_growing(self, grads, threshold, offset_mask, overlap):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.padding * cur_size
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.padding * cur_size
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()[:, :(3 + self.scale_dim)] * cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_sem_feat = self._anchor_sem_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.semantic_dim])[candidate_mask]
                new_sem_feat = scatter_max(new_sem_feat, inverse_indices.unsqueeze(1).expand(-1, new_sem_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_sem_feat": new_sem_feat,
                    "offset": new_offsets,
                }
                
                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_sem_feat = optimizable_tensors["anchor_sem_feat"]
                self._offset = optimizable_tensors["offset"]
    
    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.densification_interval * opt.success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, opt.densify_grad_threshold, offset_mask, opt.overlap)
        
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        all_opacities = self.get_all_opacities
        prune_mask = (all_opacities < opt.opacity_cull).all(dim=1)
        anchors_mask = (self.anchor_demon > opt.densification_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.any():
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
    def generate_neural_gaussians(self, viewpoint_camera = None, ape_code=-1, is_inference = False):
        anchor = self.get_anchor
        feat = self.get_anchor_feat
        semantic_feat = self.get_anchor_semantic_feat
        grid_offsets = self.get_offset
        grid_scaling = self.get_scaling

        if viewpoint_camera is not None:
            ob_view = anchor - viewpoint_camera.camera_center
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            ob_view = ob_view / (ob_dist + 1e-5)
        else:
            ob_view = torch.nn.functional.normalize(torch.rand_like(anchor), dim = 1)

        if is_inference and self.cached_gs is not None:
            xyz, semantic, opacity, scaling, rot, anchor, feat, mask = self.cached_gs

            cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]
            color = self.get_color_mlp(cat_local_view)
            color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

            color = color[mask]

            return xyz, color, semantic, opacity, scaling, rot, None, mask

        ## view-adaptive feature
        neural_opacity = self.get_opacity_mlp(feat) # [N, k]
        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0) & (~neural_opacity.isnan())
        mask = mask.view(-1)
        # select opacity 
        opacity = neural_opacity[mask]
        semantic = self.get_semantic_mlp(semantic_feat)
        semantic = semantic.reshape([anchor.shape[0]*self.n_offsets, 4]) # [mask]
        semantic = F.softmax(semantic, dim = -1)

        # get offset's cov
        scale_rot = self.get_cov_mlp(feat)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, (4 + self.scale_dim)]) # [mask]
        offsets = grid_offsets.view([-1, 3]) # [mask]

        cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        color = self.get_color_mlp(cat_local_view)
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]       

        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, semantic, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, semantic, scale_rot, offsets = masked.split([3 + self.scale_dim, 3, 3, 4, (4 + self.scale_dim), 3], dim=-1)
        
        # post-process cov
        peroffset_scaling = torch.sigmoid(scale_rot[:,:self.scale_dim])
        scaling = scaling_repeat[:,3:] * peroffset_scaling + getattr(self, "min_scale", 0.0001)
        
        rot = self.rotation_activation(scale_rot[:, self.scale_dim:])
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        if is_inference and self.cached_gs is None:
            self.cached_gs = (xyz, semantic, opacity, scaling, rot, anchor, feat, mask)

        return xyz, color, semantic, opacity, scaling, rot, None, mask

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'mlp_sem' in group['name'] or \
                'structure' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting. 
    def training_statis(self, render_pkg):
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]
        offset_selection_mask = render_pkg["selection_mask"] 

        self.anchor_demon += 1

        # update neural gaussian statis
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        grad_norm = torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'mlp_sem' in group['name'] or \
                'structure' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim + self.view_embedding).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim + self.view_embedding).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        sem_mlp = torch.jit.trace(self.mlp_sem, (torch.rand(1, self.semantic_dim + self.view_embedding).cuda()))
        sem_mlp.save(os.path.join(path, 'sem_mlp.pt'))

        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        self.mlp_sem = torch.jit.load(os.path.join(path, 'sem_mlp.pt')).cuda()

    
    def clean(self):
        del self.anchor_demon
        del self.offset_gradient_accum
        del self.offset_denom
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def init_with_learned(self, floor_points, ceiling_points, scene_center, rot):

        structure_normal = build_rotation(rot)[:, :3, 2]

        sign = torch.sign(((scene_center - floor_points) * structure_normal).sum(-1))
        structure_normal = (structure_normal * sign.unsqueeze(1)).mean(dim = 0)
        structure_normal = structure_normal / torch.norm(structure_normal)

        if floor_points.shape[0] == 0:
            floor_distance = torch.tensor(0.0, dtype=torch.float, device="cuda")
        else:
            floor_distance = (floor_points * structure_normal).sum(-1).mean()

        if ceiling_points.shape[0] == 0:
            ceiling_distance = torch.tensor(0.0, dtype=torch.float, device="cuda")
        else:
            ceiling_distance = (ceiling_points * structure_normal).sum(-1).mean()

        structure = torch.tensor([*structure_normal.cpu().numpy().tolist(), floor_distance.item(), ceiling_distance.item()], dtype=torch.float, device="cuda")
        return structure

    @torch.no_grad()
    def update_structure_prior(self, scene_center):
        self.scene_center = scene_center
        xyz, _, semantics, opacities, _, _, _, _ = self.generate_neural_gaussians()
        semantic_label = semantics.argmax(-1)

        valid_point = (opacities.squeeze() > 0.2) 
        is_floor = torch.logical_and(semantic_label == 1, valid_point)
        is_ceiling = torch.logical_and(semantic_label == 2, valid_point)

        floor_points = xyz[is_floor] 
        ceiling_points = xyz[is_ceiling]

        structure = self.init_with_ransac(floor_points, ceiling_points, scene_center)

        if not self.args.update_full:
            structure[:3] = self.structure[:3].detach()

        print("Update structure: ", structure)
        optimizable_params = self.replace_tensor_to_optimizer(structure, "structure")
        self.structure = optimizable_params["structure"]

    @torch.no_grad()
    def init_with_ransac(self, floor_points, ceiling_points, scene_center):

        floor_plane, _ = fit_plane_ransac(floor_points)
        floor_plane = floor_plane[0] / torch.norm(floor_plane[0, :3])
        floor_plane = torch.sign((floor_plane[:3] * scene_center).sum(-1)) * floor_plane
        structure_normal = floor_plane[:3]
        floor_distance = -floor_plane[3]
        if ceiling_points.shape[0] == 0:
            ceiling_distance = torch.tensor(0.0, dtype=torch.float, device="cuda")
        else:
            ceiling_distance = (ceiling_points * structure_normal).sum(-1).mean()

        structure = torch.tensor([*structure_normal.cpu().numpy().tolist(), floor_distance.item(), ceiling_distance.item()], dtype=torch.float, device="cuda")
        return structure
    
    @torch.no_grad()
    def init_structure_from_colmap(self, path):
        if os.path.exists(path):
            structure = np.loadtxt(path)
        else:
            print("No structure file found at: ", path)
            structure = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        print("Init structure: ", structure)
        return structure

    @torch.no_grad()
    def init_structure(self, scene_center = None):
        # import pdbp; pdbp.set_trace()
        self.scene_center = scene_center

        xyz, _, semantics, opacities, _, _, _, _ = self.generate_neural_gaussians()
        semantic_label = semantics.argmax(-1)

        valid_point = (opacities.squeeze() > 0.05) 
        valid_point = valid_point.reshape(-1, 1)

        is_floor = torch.logical_and(semantic_label == 1, valid_point)
        is_ceiling = torch.logical_and(semantic_label == 2, valid_point)

        floor_points = xyz[is_floor] 
        ceiling_points = xyz[is_ceiling]

        structure = self.init_with_ransac(floor_points, ceiling_points, scene_center)

        print("Init structure: ", structure)
        optimizable_params = self.replace_tensor_to_optimizer(structure, "structure")
        self.structure = optimizable_params["structure"]

    def get_num_points(self):
        return self._offset.shape[0] * self._offset.shape[1]
    
    def get_parameters(self):
        parameter_dict = {}
        for name, value in self.mlp_color.named_parameters():
            parameter_dict["mlp_color." + name] = value
        for name, value in self.mlp_opacity.named_parameters():
            parameter_dict["mlp_opacity." + name] = value
        for name, value in self.mlp_cov.named_parameters():
            parameter_dict["mlp_cov." + name] = value
        for name, value in self.mlp_sem.named_parameters():
            parameter_dict["mlp_sem." + name] = value

        parameter_dict["anchor"] = self._anchor
        parameter_dict["offset"] = self._offset
        parameter_dict["feat"] = self._anchor_feat
        parameter_dict["feat_sem"] = self._anchor_sem_feat
        parameter_dict["anchor_scaling"] = self._scaling
        parameter_dict["rot"] = self._rotation
        parameter_dict["structure"] = self.structure
        return parameter_dict