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
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp
from .general_utils import build_rotation

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    # if x_0.isnan().any() or x_1.isnan().any():
    #     import pdbp; pdbp.set_trace()

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        # import pdbp; pdbp.set_trace()
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy

class GaussianLoss(nn.Module):
    def __init__(self, opt, dataset, gaussian):
        super().__init__()
        self.opt = opt
        self.gaussian = gaussian
        self.dataset = dataset
        self.depth_prior_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=4, reduction='batch-based')

    def forward(self, iteration, render_pkg, viewpoint_cam):

        image = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        gt_image = viewpoint_cam.original_image.to(image.device)
        gt_semantic = viewpoint_cam.semantic
        if gt_semantic is not None:
            gt_semantic = gt_semantic.to(image.device)

        opt = self.opt
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        lambda_normal = opt.lambda_normal if iteration > opt.loss_normal_begin else 0.0
        lambda_dist = opt.lambda_dist if iteration > opt.loss_dist_begin else 0.0
        lambda_normal_prior = opt.lambda_normal_prior * (opt.loss_normalprior_begin - iteration) / opt.loss_normalprior_begin if iteration < opt.loss_normalprior_begin else opt.lambda_normal_prior
        lambda_semantic = opt.lambda_semantic if self.dataset.semantics and iteration > opt.loss_sem_begin else 0.0
        lambda_structure = opt.lambda_structure if iteration > opt.loss_structure_begin else 0.0
        lambda_depth_int = opt.lambda_depthprior if iteration > opt.loss_depthprior_begin else 0.0
        use_structure_2d = opt.loss_structure_2D_begin > 0 and iteration > opt.loss_structure_2D_begin

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_alpha = render_pkg["rend_alpha"]
        surf_depth = render_pkg["surf_depth"]
        loss_dict = {}

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        loss_dict["normal_loss"] = normal_loss

        dist_loss = lambda_dist * (rend_dist).mean()
        loss_dict["dist_loss"] = dist_loss
        loss_dict["l1_loss"] = Ll1.mean()

        total_loss = loss + dist_loss + normal_loss
        if lambda_normal_prior > 0 and self.dataset.normals and viewpoint_cam.normal_prior is not None:
            prior_normal = viewpoint_cam.normal_prior.to(rend_alpha.device) # * (rend_alpha).detach()
            normal_prior_error = ((1 - F.cosine_similarity(prior_normal, rend_normal, dim=0))).mean() + \
                                 ((1 - F.cosine_similarity(prior_normal, surf_normal, dim=0))).mean() #+ \

            normal_prior_loss = lambda_normal_prior * normal_prior_error
            loss_dict["normal_prior_loss"] = normal_prior_error
            total_loss += normal_prior_loss

        if lambda_depth_int > 0 and self.dataset.depths:
            depth_prior_loss = self.depth_prior_loss(surf_depth, viewpoint_cam.depth.to(surf_depth.device), viewpoint_cam.mask.to(surf_depth.device))
            loss_dict["depth_prior_loss"] = depth_prior_loss
            total_loss += lambda_depth_int * depth_prior_loss
        
        if lambda_semantic > 0:
            semantic_loss = self.get_semantic_loss(gt_semantic, render_pkg["rend_semantic"])
            loss_dict["semantic_loss"] = semantic_loss
            total_loss += lambda_semantic * semantic_loss
        
        if lambda_structure > 0:
            structure_loss, structure_loss_dict = self.get_structure_loss(render_pkg["gaussian_attrs"], visibility_filter)
            loss_dict.update(structure_loss_dict)
            total_loss += lambda_structure * structure_loss

            if use_structure_2d:
                structure_loss_2d, structure_loss_dict_2d = self.get_structure_loss_2d(
                    render_pkg["gaussian_attrs"]["structure"], render_pkg["rend_semantic"], 
                    rend_normal, surf_normal
                )
                loss_dict.update(structure_loss_dict_2d)
                total_loss += lambda_structure * structure_loss_2d
        

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
    
    def get_structure_loss_2d(self, structure, rend_semantic, rend_normal, surf_normal):
        loss_dict = {}
        wall_prob = rend_semantic[0]
        floor_prob = rend_semantic[1] + rend_semantic[2]
        structure_normal = structure["structure_normal"][:3]
        rend_error = torch.abs((rend_normal * structure_normal.reshape(3, 1, 1)).sum(0))
        surf_error = torch.abs((surf_normal * structure_normal.reshape(3, 1, 1)).sum(0))

        wall_loss = (rend_error * wall_prob).mean() + (surf_error * wall_prob)[1:-1, 1:-1].mean()
        floor_loss = ((1 - rend_error) * floor_prob).mean() + ((1 - surf_error) * floor_prob)[1:-1, 1:-1].mean()

        loss_dict["struct_2d_loss_wall"] = wall_loss
        loss_dict["struct_2d_loss_floor"] = floor_loss
        return wall_loss * self.opt.lambda_wall + floor_loss * self.opt.lambda_floor, loss_dict

    def get_structure_loss(self, attrs, visibility_filter):
        structure_loss = 0
        loss_dict = {}
        xyz = attrs["xyz"][visibility_filter]
        semantic = attrs["semantic"][visibility_filter]
        semantic_label = semantic.argmax(dim=-1)
        structure_normal = attrs["structure"]["structure_normal"][:3]
        floor_structure = attrs["structure"]["structure_distance"][0]
        ceiling_structure = attrs["structure"]["structure_distance"][1]
        opacity_mask = attrs["opacity"][visibility_filter].squeeze(-1) > 0.05

        rot = attrs["rot"][visibility_filter]
        normal = build_rotation(rot)[:, :, 2]

        with torch.no_grad():
            is_wall = (semantic_label == 0) & opacity_mask
            is_floor = (semantic_label == 1) & opacity_mask
            is_ceiling = (semantic_label == 2) & opacity_mask

        ceiling_prob = semantic[is_ceiling, 2]
        floor_prob = semantic[is_floor, 1]
        wall_prob = semantic[is_wall, 0]

        if is_ceiling.any():
            normal_cos_ceiling = (normal[is_ceiling] * structure_normal).sum(-1)
            ceiling_normal_loss = ((1 - torch.abs(normal_cos_ceiling)) * ceiling_prob).mean()
            loss_dict["ceiling_normal_loss"] = ceiling_normal_loss
            structure_loss += ceiling_normal_loss

            ceiling_structure_loss = torch.abs((xyz[is_ceiling] * structure_normal).sum(-1) - ceiling_structure) * ceiling_prob
            ceiling_structure_loss = ceiling_structure_loss.mean()

            loss_dict["ceiling_structure_loss"] = ceiling_structure_loss
            structure_loss += ceiling_structure_loss
        
        if is_floor.any():
            normal_cos_floor = (normal[is_floor] * structure_normal).sum(-1)
            floor_normal_loss = ((1 - torch.abs(normal_cos_floor)) * floor_prob).mean()
            loss_dict["floor_normal_loss"] = floor_normal_loss
            structure_loss += floor_normal_loss

            floor_structure_loss = torch.abs((xyz[is_floor] * structure_normal).sum(-1) - floor_structure) * floor_prob
            floor_structure_loss = floor_structure_loss.mean()
            loss_dict["floor_structure_loss"] = floor_structure_loss
            structure_loss += floor_structure_loss
        
        if is_wall.any():
            wall_normal_loss = torch.abs((normal[is_wall] * structure_normal).sum(-1))
            wall_normal_loss = wall_normal_loss.mean()
            loss_dict["wall_normal_loss"] = wall_normal_loss
            structure_loss += wall_normal_loss

        loss_dict["floor"] = floor_structure
        loss_dict["ceiling"] = ceiling_structure
        
        loss_dict["structure_loss"] = structure_loss
        return structure_loss, loss_dict
    
    def get_semantic_loss(self, gt_semantic, rend_semantic):
        rend_semantic = torch.log(torch.clamp(rend_semantic.unsqueeze(0), min = 1e-6, max=1-1e-6))
        semantic_loss = F.cross_entropy(
            rend_semantic,
            gt_semantic[:4].unsqueeze(0),
            reduction='none'
        ) * gt_semantic[4:]
        semantic_loss = semantic_loss.mean()
        return semantic_loss