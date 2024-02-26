import torch
from sdf import SDFLoss
import torch.nn.functional as torch_f
from pytorch3d.structures.meshes import Meshes
import matplotlib.pyplot as plt

from obman_net.mano_train.networks.branches.manobranch import ManoLoss
from obman_net.mano_train.networks.branches import atlasutils
from loss.Render import Render

class PoseHOI_Loss:
    def __init__(self):
        self.mask_weight = 1
        self.mesh_weight = 0.0167
        self.param_weight = 0.167
        self.hand_weight = 1
        self.contact_weight = 1
        self.chamfer_weight = 0.167
        self.mano_loss = ManoLoss(
            lambda_verts=0.167,
            lambda_joints3d=0.167,
            lambda_shape=0.167,
            lambda_pose_reg=0.167,
            lambda_pca=0,
        )

        self.render = Render()
        self.c_loss = atlasutils.ChamferLoss()
    
    def align_verts(self, prior_verts):
        # prior_verts: [bs, N, 3]
        centered_verts = prior_verts - prior_verts.mean(1).unsqueeze(1)
        scale = torch.norm(centered_verts, p=2, dim=2).max(1)[0]
        scaled_verts = centered_verts / scale.unsqueeze(1).unsqueeze(1)
        return scaled_verts

    def center_verts(self, prior_verts):
        centered_verts = prior_verts - prior_verts.mean(1).unsqueeze(1)
        return centered_verts

    def gt_align(self, prior_verts, gt_verts):
        center_gt = gt_verts - gt_verts.mean(1).unsqueeze(1)
        scale_gt = torch.norm(center_gt, p=2, dim=2).max(1)[0]
        center_prior = prior_verts - prior_verts.mean(1).unsqueeze(1)
        scale_prior = torch.norm(center_prior, p=2, dim=2).max(1)[0]

        gt_prior = center_prior / scale_prior.unsqueeze(1).unsqueeze(1) * scale_gt.unsqueeze(1).unsqueeze(1)

        return gt_prior

    def compute_mesh_loss(self, prior_verts, gt_prior_verts):
        # a_prior_verts = self.align_verts(prior_verts) * 1000
        # a_gt_prior_verts = self.align_verts(gt_prior_verts) * 1000
        center_prior_verts = self.center_verts(prior_verts)
        center_gt_prior_verts = self.center_verts(gt_prior_verts)
        mesh_loss = torch_f.mse_loss(center_prior_verts, center_gt_prior_verts)
        return mesh_loss
    
    def compute_pc_loss(self, pred_verts, gt_verts):
        loss_1, loss_2 = self.c_loss(pred_verts, gt_verts)
        chamfer_distance = torch.mean(loss_1 + loss_2)
        return chamfer_distance
    
    def compute_dense_pc_loss(self, pred_verts, gt_verts):
        loss_1, loss_2 = self.c_loss.calc_dcd(pred_verts, gt_verts)
        dense_chamfer_distance = torch.mean(loss_1 + loss_2)
        return dense_chamfer_distance

    def compute_render_loss(self, prior_verts, prior_faces, gt_mask, device):
        # prior_verts: [bs, 642, 3]
        gt_mask = gt_mask.float().cuda()
        prior_verts = prior_verts.cuda()
        prior_faces = prior_faces.cuda()
        aligned_verts = self.align_verts(prior_verts)
        meshes = Meshes(aligned_verts, prior_faces)
        pred_mask = self.render.rend(meshes) # [bs, 128, 128]
        # mask_loss = torch_f.mse_loss(pred_mask, gt_mask)
        mask_loss = ((pred_mask - gt_mask) ** 2).mean() * 1000
        return mask_loss, pred_mask

    def compute_param_loss(self, pred_verts, gt_verts):
        pred_trans = pred_verts.mean(1)
        gt_trans = gt_verts.mean(1)
        centered_pred_verts = pred_verts - pred_trans.unsqueeze(1)
        centered_gt_verts = gt_verts - gt_trans.unsqueeze(1)

        pred_scale = torch.norm(centered_pred_verts, 2, 2).max(1)[0]
        gt_scale = torch.norm(centered_gt_verts, 2, 2).max(1)[0]

        trans_loss = torch_f.mse_loss(pred_trans, gt_trans)
        scale_loss = torch_f.mse_loss(pred_scale, gt_scale)

        # c1, c2, _, _ = self.c_loss.calc_cdc(centered_pred_verts, centered_gt_verts)
        # chamfer_loss = torch.mean(c1 + c2)
        
        param_loss = trans_loss + scale_loss

        return trans_loss, scale_loss, param_loss

    def compute_mano_handloss(self, pred_handverts, pred_handjoint, 
                         target_handverts, target_handjoint,
                         pred_shape, pred_pose):
        hand_loss = self.mano_loss.compute_myloss(
            pred_handverts, pred_handjoint,
            target_handverts, target_handjoint,
            pred_shape, pred_pose
        )
        return hand_loss
    
    def compute_contact_loss(self, batch_size, hand_verts, obj_verts, hand_faces, obj_faces):
        bs = batch_size
        obj_faces = obj_faces[0].numpy() 

        hand_verts = hand_verts.unsqueeze(dim=1) # (bs, 1, 778, 3)
        obj_verts = obj_verts.unsqueeze(dim=1) # (bs, 1, 642, 3)

        sdf_loss = SDFLoss(hand_faces, obj_faces, robustifier=None).cuda()

        losses = sdf_loss(
            hand_verts, obj_verts, return_per_vert_loss=True, return_origin_scale_loss=True)
        losses = losses.reshape(bs, 1)

        loss = torch.mean(losses)
        return loss
    
    def compute_loss(self, prior_verts, prior_faces, hand_faces,
                     trans, scale,
                     pred_handverts, pred_handjoint, pred_shape, pred_pose,
                     gt_mask, gt_verts, gt_prior_verts, gt_handverts, gt_handjoint,
                     device
                    ):
        bs = prior_verts.shape[0]
        pred_verts = prior_verts * scale.unsqueeze(1) + trans.unsqueeze(1)
        # in order to better compute collision loss to refine rotation block, align hand and obj 
        gt_trans = gt_verts.mean(1)
        ho_verts = prior_verts * scale.unsqueeze(1) + gt_trans.unsqueeze(1)
        gt_prior_verts = self.gt_align(gt_prior_verts, gt_verts) + gt_trans.unsqueeze(1)
        # mask_loss, pred_mask = self.compute_render_loss(pred_verts, prior_faces, gt_mask, device)
        trans_loss, scale_loss, param_loss = self.compute_param_loss(pred_verts, gt_verts)
        hand_loss = self.compute_mano_handloss(pred_handverts, pred_handjoint, gt_handverts, gt_handjoint,
                                               pred_shape, pred_pose)
        contact_loss = self.compute_contact_loss(bs, pred_handverts, ho_verts, hand_faces, prior_faces)
        mesh_loss = self.compute_mesh_loss(prior_verts, gt_prior_verts)

        total_loss = (self.mesh_weight * mesh_loss
                    + self.param_weight * param_loss
                    + self.hand_weight * hand_loss
                    + self.contact_weight * contact_loss
        )

        return total_loss, mesh_loss, param_loss, hand_loss, contact_loss, trans_loss, scale_loss
    
    def compute_refine_loss(self, pred_verts, pred_faces, hand_faces,
                            pred_handverts, pred_handjoint, pred_shape, pred_pose,
                            gt_verts, gt_handverts, gt_handjoint):
        bs = pred_verts.shape[0]
        chamfer_loss = self.compute_pc_loss(pred_verts, gt_verts)
        # contact_loss = self.compute_contact_loss(bs, pred_handverts, pred_verts, hand_faces, pred_faces)
        contact_loss = torch.tensor([0]).cuda()
        trans_loss, scale_loss, param_loss = self.compute_param_loss(pred_verts, gt_verts)
        hand_loss = self.compute_mano_handloss(pred_handverts, pred_handjoint, gt_handverts, gt_handjoint,
                                               pred_shape, pred_pose)

        total_loss = (self.chamfer_weight * chamfer_loss
                    + self.param_weight * param_loss
                    + self.hand_weight * hand_loss
                    + self.contact_weight * contact_loss
        )
        return total_loss, chamfer_loss, param_loss, hand_loss, contact_loss, trans_loss, scale_loss

    
    # no hand loss
    def compute_loss_ho3d_eval(self, prior_verts, prior_faces, hand_faces,
                     trans, scale,
                     pred_handverts, pred_handjoint, pred_shape, pred_pose,
                     gt_mask, gt_verts, gt_prior_verts, device
                    ):
        bs = prior_verts.shape[0]
        # in order to better compute collision loss to refine rotation block, align hand and obj 
        gt_trans = gt_verts.mean(1)
        pred_verts = prior_verts * scale.unsqueeze(1) + trans.unsqueeze(1)
        ho_verts = prior_verts * scale.unsqueeze(1) + gt_trans.unsqueeze(1)
        # mesh_loss, pred_mask = self.compute_render_loss(pred_verts, prior_faces, gt_mask, device)
        trans_loss, scale_loss, param_loss = self.compute_param_loss(pred_verts, gt_verts)
        mesh_loss = self.compute_mesh_loss(prior_verts, gt_prior_verts)
        contact_loss = 0

        total_loss = (self.mesh_weight * mesh_loss
                    + self.param_weight * param_loss
                    + self.contact_weight * contact_loss
        )

        return total_loss, mesh_loss, param_loss, contact_loss, trans_loss, scale_loss

    # no hand loss
    def compute_refine_loss_ho3d_eval(self, pred_verts, pred_faces, hand_faces,
                     pred_handverts, pred_handjoint, pred_shape, pred_pose,
                     gt_verts
                    ):
        bs = pred_verts.shape[0]
        chamfer_loss = self.compute_pc_loss(pred_verts, gt_verts)
        # contact_loss = self.compute_contact_loss(bs, pred_handverts, pred_verts, hand_faces, pred_faces)
        contact_loss = torch.tensor([0]).cuda()
        trans_loss, scale_loss, param_loss = self.compute_param_loss(pred_verts, gt_verts)

        total_loss = (self.chamfer_weight * chamfer_loss
                    + self.param_weight * param_loss
                    + self.contact_weight * contact_loss
        )
        return total_loss, chamfer_loss, param_loss, contact_loss, trans_loss, scale_loss