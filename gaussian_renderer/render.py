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

import torch
import math
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier = 1.0, *args, **kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_inference = kwargs.get('inference', False)

    xyz, color, semantic, opacity, scaling, rot, _, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, is_inference=is_inference)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if scaling.shape[-1] == 3:
        scaling = scaling[..., :2]
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        detach_sem = pipe.detach_sem
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = scaling
    rotations = rot

    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = color

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantics = semantic,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "selection_mask": selection_mask,
        "opacity": opacity,
    }

    # additional regularizations
    render_alpha = torch.clamp(allmap[1:2], min=1e-5)

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    normal_norm = torch.norm(render_normal, p=2, dim = 0, keepdim=True) + 1e-5
    view_normal = render_normal / normal_norm
    render_normal = (view_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    rend_semantic = allmap[7:11] 
    rend_semantic = F.normalize(rend_semantic, p=1, dim=0, eps=1e-8)

    rend_plane = allmap[-1:] / normal_norm.detach()

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)

    surf_normal_median = depth_to_normal(viewpoint_camera, render_depth_median)
    surf_normal_median = surf_normal_median.permute(2,0,1)

    gaussian_attrs = {
        "xyz": xyz,
        "semantic": semantic[:, :4],
        "rot": rot,
        "scaling": scaling,
        "structure": pc.get_structure,
        "opacity": opacity,
    }

    # import pdbp; pdbp.set_trace()
    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'rend_depth_median': render_depth_median,
            'rend_semantic': rend_semantic,
            'view_normal': view_normal,
            'rend_plane': rend_plane,

            'rend_depth': render_depth_expected,
            'surf_depth': surf_depth,

            'surf_normal': surf_normal,
            'surf_normal_median': surf_normal_median,

            # "contrib": contrib,
            "gaussian_attrs": gaussian_attrs
    })

    return rets
