import torch
import torch.nn as nn
import numpy as np
from sdf import SDFLoss
import torch.nn.functional as torch_f
import warnings

from hand_pipline.edge_loss import edge_length_loss
from obman_net.mano_train.networks.branches import atlasutils
from obman_net.mano_train.networks.branches.atlasbranch import AtlasLoss
from obman_net.mano_train.networks.branches.manobranch import ManoLoss
from obman_net.mano_train.networks.branches.contactloss import compute_contact_loss
from loss.posehoi_loss import PoseHOI_Loss

def get_loss(args):
    if args["model_name"] == "TemplateHOI":
        return Ho_Loss()
    elif args["model_name"] == "Obman":
        return Obman_Loss()
    elif args["model_name"] == "PoseBlock":
        return Prior_loss()
    elif args["model_name"] == "PoseHOI":
        return PoseHOI_Loss()

class Ho_Loss():
    def __init__(self):
        self.mano_loss = ManoLoss(
            lambda_verts=0.167,
            lambda_joints3d=0.167,
            lambda_shape=0.167,
            lambda_pose_reg=0.167,
            lambda_pca=0,
        )
        self.atlas_loss_model = AtlasLoss(
            atlas_loss="chamfer",
            lambda_atlas=0,
            final_lambda_atlas=0,
            trans_weight=0.167,
            scale_weight=0.167,
            center_weight=0.167,
            reg_weight=0,
            edge_regul_lambda=None,
            lambda_laplacian=0,
            laplacian_faces=None,
            laplacian_verts=None,
        )
        self.prior_loss_model = AtlasLoss(
            atlas_loss="chamfer",
            lambda_atlas=0,
            final_lambda_atlas=0,
            trans_weight=0,
            scale_weight=0,
            center_weight=0.167,
            edge_regul_lambda=None,
            lambda_laplacian=0,
            laplacian_faces=None,
            laplacian_verts=None,
        )

    def compute_handloss(self, pred_hand_verts, pred_hand_joint, target_hand_verts, target_hand_joint, hand_tmp):
        hand_verts_loss = 0
        hand_edge_loss = 0
        hand_pose_loss = nn.L1Loss()(pred_hand_joint, target_hand_joint)

        for idx in range(4):
            faces_right = hand_tmp['face'][idx].astype(np.int16)
            hand_verts_loss += nn.L1Loss()(pred_hand_verts[3-idx], target_hand_verts[idx])
            hand_edge_loss += edge_length_loss(pred_hand_verts[3-idx], target_hand_verts[idx], faces_right, is_valid=None)

        hand_loss = hand_verts_loss + hand_edge_loss*0.1 + hand_pose_loss

        return hand_loss

    def compute_mano_handloss(self, pred_handverts, pred_handjoint, 
                         target_handverts, target_handjoint,
                         pred_shape, pred_pose):
        hand_loss = self.mano_loss.compute_myloss(
            pred_handverts, pred_handjoint,
            target_handverts, target_handjoint,
            pred_shape, pred_pose
        )
        return hand_loss

    def compute_object_loss(self, pred_verts, pred_trans, pred_scale, obj_point_gt, mask_points=None, delta=None):
        atlas_loss, _, _, reg_loss, _ = self.atlas_loss_model.my_compute_loss(
            pred_verts=pred_verts, pred_trans=pred_trans, pred_scale=pred_scale, pred_rot=None, gt_rot=None,
            target=obj_point_gt, mask_points=None, delta=delta
        )
        return atlas_loss, reg_loss

    def compute_contact_loss(self, batch_size, hand_verts, obj_verts, hand_faces, obj_faces):
        bs = batch_size

        hand_verts = hand_verts.unsqueeze(dim=1) # (bs, 1, 778, 3)
        obj_verts = obj_verts.unsqueeze(dim=1) # (bs, 1, 642, 3)

        sdf_loss = SDFLoss(hand_faces, obj_faces, robustifier=None).cuda()

        losses = sdf_loss(
            hand_verts, obj_verts, return_per_vert_loss=True, return_origin_scale_loss=True)
        losses = losses.reshape(bs, 1)

        loss = torch.mean(losses)
        return loss

    def compute_loss(self, args, hand_tmp,
                     pred_objverts, pred_objtrans, pred_objscale, gt_objverts,
                     pred_handverts, pred_handjoints, pred_handshape, pred_handpose, gt_handverts, gt_handjoints,
                     hand_faces, obj_faces,
                     delta, mask_points=None
                    ):
        warnings.filterwarnings("ignore")
        atlas_loss, reg_loss = self.compute_object_loss(
            pred_objverts, pred_objtrans, pred_objscale, gt_objverts, mask_points, delta
        )

        if args["dataset"] == "ho3d" and args["mode"] != "train":
            hand_loss = 0
            contact_loss = 0
        else:
            hand_loss = self.compute_mano_handloss(
                pred_handverts, pred_handjoints, gt_handverts, gt_handjoints,
                pred_handshape, pred_handpose
            )
            # hand_loss = self.compute_handloss(
            #     pred_handverts, pred_handjoints, gt_handverts, gt_handjoints, hand_tmp
            # )
            # contact_loss = self.compute_contact_loss(
            #     args["batch_size"], pred_handverts, pred_objverts, hand_faces, obj_faces
            # )
            contact_loss = 0

        total_loss = atlas_loss + hand_loss + contact_loss

        return atlas_loss, hand_loss, contact_loss, total_loss, reg_loss

    def compute_prior_loss(self, args, hand_tmp,
            pred_objverts, pred_objtrans, pred_objscale, gt_objverts,
            pred_handverts, pred_handjoints, hand_shape, hand_pose, gt_handverts, gt_handjoints,
            hand_faces, obj_faces,
            mask_points, prior_pc, delta
        ):
        p_loss, _, _, _, _ = self.prior_loss_model.my_compute_loss(
            pred_verts=prior_pc, pred_trans=None, pred_scale=None, pred_rot=None, gt_rot=None,
            target=gt_objverts, mask_points=None, delta=None
        )
        atlas_loss, hand_loss, contact_loss, standard_loss, reg_loss = self.compute_loss(args, hand_tmp,
            pred_objverts, pred_objtrans, pred_objscale, gt_objverts,
            pred_handverts, pred_handjoints, hand_shape, hand_pose, gt_handverts, gt_handjoints,
            hand_faces, obj_faces, delta, mask_points
        )
        total_loss = p_loss + standard_loss
        return atlas_loss, hand_loss, contact_loss, standard_loss, total_loss, reg_loss

class Obman_Loss:
    def __init__(self):
        self.mano_loss = ManoLoss(
            lambda_verts=0.167,
            lambda_joints3d=0.167,
            lambda_shape=0.167,
            lambda_pose_reg=0.167,
            lambda_pca=0,
        )
        self.atlas_loss = AtlasLoss(
            atlas_loss="chamfer",
            lambda_atlas=0,
            final_lambda_atlas=0.167,
            trans_weight=0.167,
            scale_weight=0.167,
            edge_regul_lambda=None,
            lambda_laplacian=0,
            laplacian_faces=None,
            laplacian_verts=None,
        )
        self.contact_target = "all"
        self.contact_zones = "all"
        self.contact_lambda = 0
        self.contact_thresh = 10
        self.contact_mode = "dist_tanh"
        self.collision_lambda = 0
        self.collision_thresh = 20
        self.collision_mode = "dist_tanh"
        if self.contact_lambda or self.collision_lambda:
            self.need_collisions = True
        else:
            self.need_collisions = False
    
    def compute_handloss(self, pred_handverts, pred_handjoint, 
                         target_handverts, target_handjoint,
                         pred_shape, pred_pose):
        hand_loss = self.mano_loss.compute_myloss(
            pred_handverts, pred_handjoint,
            target_handverts, target_handjoint,
            pred_shape, pred_pose
        )
        return hand_loss

    def compute_object_loss(self, pred_verts, pred_trans, pred_scale, obj_point_gt):
        atlas_loss, _, _, _, _ = self.atlas_loss.my_compute_loss(
            pred_verts, pred_trans, pred_scale, None, None, obj_point_gt
        )
        return atlas_loss
    
    def compute_contact_loss(self, handverts, handfaces, objverts, objfaces):
        attr_loss,penetr_loss, _, _ = compute_contact_loss(
            hand_verts_pt=handverts,
            hand_faces=handfaces,
            obj_verts_pt=objverts,
            obj_faces=objfaces,
            contact_thresh=self.contact_thresh,
            contact_mode=self.contact_mode,
            collision_thresh=self.collision_thresh,
            collision_mode=self.collision_mode,
            contact_target=self.contact_target,
            contact_zones=self.contact_zones,
        )
        contact_loss = (
            self.contact_lambda * attr_loss
            + self.collision_lambda * penetr_loss
        )
        return contact_loss.item()

    def compute_loss(self, 
                     pred_objverts, pred_objtrans, pred_objscale, gt_objverts,
                     pred_handverts, pred_handjoints, pred_handshape, pred_handpose, 
                     gt_handverts, gt_handjoints, 
                     hand_faces, obj_faces,
                    ):
        hand_loss = self.compute_handloss(
            pred_handverts, pred_handjoints, gt_handverts, gt_handjoints,
            pred_handshape, pred_handpose
        )
        obj_loss = self.compute_object_loss(
            pred_objverts, pred_objtrans, pred_objscale, gt_objverts
        )
        contact_loss = self.compute_contact_loss(
            pred_handverts, hand_faces, pred_objverts, obj_faces
        )
        total_loss = hand_loss + obj_loss + contact_loss

        return obj_loss, hand_loss, contact_loss, total_loss

class Prior_loss():
    def __init__(self):
        self.atlas_loss_model = AtlasLoss(
            atlas_loss="chamfer",
            lambda_atlas=0,
            center_weight=0.167,
            final_lambda_atlas=0,
            trans_weight=0,
            scale_weight=0,
            rot_weight=0,
            edge_regul_lambda=None,
            lambda_laplacian=0,
            laplacian_faces=None,
            laplacian_verts=None,
        )

        self.auxiliary_loss_model = AtlasLoss(
            atlas_loss="chamfer",
            lambda_atlas=0,
            center_weight=0.167,
            final_lambda_atlas=0,
            trans_weight=0,
            scale_weight=0,
            rot_weight=1,
            edge_regul_lambda=None,
            lambda_laplacian=0,
            laplacian_faces=None,
            laplacian_verts=None,
        )
    
    def compute_loss(self, pred_verts, obj_point_gt):
        atlas_loss, _, _, _, rot_loss = self.atlas_loss_model.my_compute_loss(
            pred_verts=pred_verts, pred_trans=None, pred_scale=None, pred_rot=None, gt_rot=None, target=obj_point_gt
        )
        return atlas_loss, rot_loss
    
    def compute_auxiliary_loss(self, pred_verts, pred_rot, gt_rot, obj_point_gt):
        atlas_loss, _, _, _, rot_loss = self.auxiliary_loss_model.my_compute_loss(
            pred_verts=pred_verts, pred_trans=None, pred_scale=None, pred_rot=pred_rot, gt_rot=gt_rot, target=obj_point_gt
        )
        return atlas_loss, rot_loss