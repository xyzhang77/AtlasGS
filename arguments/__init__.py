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

from configargparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        self.normals = "normals"
        self.depths = "depths"
        self.semantics = "semantics"
        self.update_full = False

        self.feat_dim = 32
        self.view_dim = 3
        self.padding = 0.0
        self.appearance_dim = 0
        self.semantic_dim = 32
        self.n_offsets = 10
        self.voxel_size = 0.01
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.alpha = 0.99
        self.downsample_factor = 1.0
        self.use_colmap = True

        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.prefilter = False
        self.depth_ratio = 0.0
        self.debug = False
        self.detach_sem = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 40_000

        
        self.offset_lr_init =  0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 40_000

        self.opacity_lr = 0.05
        self.feature_lr = 0.0075
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 40_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 40_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 40_000

        self.mlp_sem_lr_init = 0.008
        self.mlp_sem_lr_final = 0.00005
        self.mlp_sem_lr_delay_mult = 0.01
        self.mlp_sem_lr_max_steps = 40_000
        
        self.structure_lr_init =  0.005
        self.structure_lr_final = 0.001
        self.structure_lr_delay_mult = 0.1

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 100.0
        self.lambda_normal = 0.05
        self.lambda_normal_prior = 0.1
        self.lambda_depthprior = 0.25
        self.lambda_semantic = 1.0
        self.lambda_structure = 0.1
        self.lambda_wall = 1.0
        self.lambda_floor = 1.0

        self.opacity_cull = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 1500
        self.start_stat = 500
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.0002

        self.overlap = False
        self.success_threshold = 0.5

        self.later_progressive = False
        
        self.loss_normal_begin = 7000
        self.loss_dist_begin = 3000
        self.loss_depthprior_begin = 500
        self.loss_depth_tv_begin = 7000
        self.loss_sem_begin = 0

        self.loss_structure_begin = 7000
        self.loss_structure_2D_begin = 20000
        self.loss_normalprior_begin = 7000

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)