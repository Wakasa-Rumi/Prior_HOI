import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from sdf import SDFLoss
# from .transform_utils import batch_rodrigues

def ho_collision_loss(batch_size, hand_verts, obj_verts, hand_faces, obj_faces):
    bs = batch_size

    hand_verts = hand_verts.unsqueeze(dim=1) # (bs, 1, 778, 3)
    obj_verts = obj_verts.unsqueeze(dim=1) # (bs, 1, 642, 3)
    # verts = torch.cat([hand_verts, obj_verts], dim=1)

    sdf_loss = SDFLoss(hand_faces, obj_faces[0], robustifier=None).cuda()

    losses = sdf_loss(
        hand_verts, obj_verts, return_per_vert_loss=True, return_origin_scale_loss=True)
    losses = losses.reshape(bs, 1)

    loss = torch.mean(losses)
    return loss