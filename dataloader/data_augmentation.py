import torch
import math
import numpy as np
from random import random
import pytorch3d.transforms.rotation_conversions as rot_cvt
from preprocess.project_template import project_template

def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat

def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat

def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot

def prior_adjust(prior, scale=0.4):
    # prior [1024, 3]
    center = prior.mean(0)
    centered_prior = prior - center

    draw = (torch.rand(1, 3) - 0.5) * scale + 1
    re_scaled_prior = prior * draw
    scale = torch.norm(re_scaled_prior, 2, 1).max(0)[0]
    re_prior = re_scaled_prior / scale

    return re_scaled_prior

def generate_randn_axis():
    axis = torch.rand(3) * 2 * math.pi
    return axis_angle_t_to_matrix(axis)

def generate_noise_axis():
    big_axis = torch.randint(0, 4, (1, 3)).squeeze(0).float() * math.pi / 2
    small_axis = (torch.rand(3) - 0.5) * math.pi / 10
    axis = big_axis + small_axis
    return axis_angle_t_to_matrix(axis)

def initial_temp_rot(template_1024, template_4096):
    new_rot = generate_randn_axis()[:3,:3]
    # new_rot = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rot_template = np.matmul(template_4096, new_rot.T)
    rot_mask = project_template(rot_template, 2) # get xz mask
    out_rot_template = np.matmul(template_1024, new_rot.T)
    return out_rot_template, rot_mask, new_rot