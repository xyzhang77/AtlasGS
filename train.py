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
import sys
import uuid

os.environ["MKL_NUM_THREADS"] = "8" # noqa
os.environ["OMP_NUM_THREADS"] = "8" # noqa 

import torch
import yaml

from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from configargparse import ArgumentParser, Namespace
from random import randint
from tqdm import tqdm
TENSORBOARD_FOUND = True

from utils.loss_utils import l1_loss
from utils.loss_utils import GaussianLoss
from gaussian_renderer.render import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    testing_iterations.extend([opt.densify_from_iter, opt.iterations])
    saving_iterations.extend([opt.densify_from_iter, opt.iterations])

    with open(os.path.join(dataset.model_path, "config.yaml"), 'w') as f:
        yaml.dump(vars(args), f, indent=4)
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    loss_fn = GaussianLoss(opt, dataset, gaussians)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_dict = {}
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        lr_dict_log = gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, iteration = iteration, render_mode = "RGB+ED")
        total_loss, loss_dict = loss_fn(iteration, render_pkg, viewpoint_cam)
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            for key, value in loss_dict.items():
                if key in ema_loss_dict:
                    ema_loss_dict[key] = 0.4 * value + 0.6 * ema_loss_dict[key]
                else:
                    ema_loss_dict[key] = value

            if iteration % 10 == 0:
                log_loss_dict = {
                    "Loss": "{:.5f}".format(ema_loss_dict["total_loss"]),
                    "distort": "{:.5f}".format(ema_loss_dict["dist_loss"]),
                    "normal": "{:.5f}".format(ema_loss_dict["normal_loss"]),
                    "Points": "{}".format(sum(gaussians.get_offset.shape[:-1])),
                    "prior": "{:.5f}".format(ema_loss_dict.get("normal_prior_loss", 0)),
                }
                progress_bar.set_postfix(log_loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            torch.cuda.synchronize()
            training_report(tb_writer, iteration, loss_dict, lr_dict_log, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background))

            if opt.lambda_structure > 0 and iteration == opt.loss_structure_begin:
                scene_center = scene.scene_center
                gaussians.update_structure_prior(scene_center)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration >  opt.start_stat:
                gaussians.training_statis(render_pkg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.run_densify(iteration, opt)

            elif iteration == opt.densify_until_iter:
                gaussians.clean()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, loss_dict, lr_dict_log, elapsed, testing_iterations, scene : Scene, renderArgs):
    if tb_writer:
        for key, value in loss_dict.items():
            # print(key, value)
            tb_writer.add_scalar('train_loss_patches/' + key, value, iteration)
        for key, value in lr_dict_log.items():
            tb_writer.add_scalar('lr/' + key, value, iteration)
            
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_offset.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = render(viewpoint, scene.gaussians, *renderArgs, iteration=iteration, render_mode="RGB+ED")
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap, map_semantic_to_color
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        try:
                            rend_semantic = render_pkg["rend_semantic"].argmax(0) + 1
                            rend_semantic = map_semantic_to_color(rend_semantic)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_semantic".format(viewpoint.image_name), rend_semantic[None], global_step=iteration)
                        except: 
                            pass

                        try:
                            rend_plane = render_pkg["rend_plane"]
                            rend_plane = rend_plane / rend_plane.max()
                            rend_plane = colormap(rend_plane.cpu().numpy()[0], cmap='turbo')
                        except:
                            pass
                        tb_writer.add_images(config['name'] + "_view_{}/rend_plane".format(viewpoint.image_name), rend_plane[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        try:
                            prior_normal = viewpoint._normal * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/prior_normal".format(viewpoint.image_name), prior_normal[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--config', type=str, default=None, is_config_file=True, help='config file path')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args)

    # All done
    print("\nTraining complete.")
