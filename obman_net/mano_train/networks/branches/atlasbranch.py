import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f
import trimesh
import math

from models.attention_utils.AttentionBlock import MultiAttentionLayer
from obman_net.handobjectdatasets.queries import TransQueries
from models.attention_utils.Projecter import icosphere_2d

from obman_net.mano_train.networks.branches import atlasutils
from obman_net.mano_train.networks.branches.laplacianloss import LaplacianLoss

def point_normalize(pc): # [n, 3]
    center = pc.mean(0)
    centered_pc = pc - center
    scale = torch.norm(centered_pc, p=2, dim=1).max(0)[0]

    nor_pc = centered_pc / scale
    # center = (pc.max(0)[0] + pc.min(0)[0]) / 2
    # centered_pc = pc - center
    # radius = np.linalg.norm(centered_pc, 2, 1).max()
    # nor_pc = centered_pc / radius
    print("scale: ", scale)

    return nor_pc, centered_pc, scale

class AtlasBranch(nn.Module):
    def __init__(
        self,
        use_residual=True,
        use_unet=True,
        mode="sphere",
        points_nb=600,
        bottleneck_size=1024,
        use_tanh=False,
        inference_ico_divisions=3,
        predict_trans=False,
        predict_scale=False,
        predict_draw=False,
        out_factor=200,
        separate_encoder=False,
    ):
        super(AtlasBranch, self).__init__()
        self.mode = mode
        self.points_nb = points_nb
        self.bottleneck_size = bottleneck_size
        self.separate_encoder = separate_encoder
        self.use_residual = use_residual
        self.use_unet = use_unet
        if self.use_residual:
            self.decoder = atlasutils.PointGenConResidual(
                bottleneck_size=3 + self.bottleneck_size, out_factor=out_factor
            )
        else:
            self.decoder = atlasutils.PointGenCon_Origin(
                bottleneck_size=515,
                out_factor=out_factor,
                use_tanh=use_tanh,
            )      
              
        self.predict_trans = predict_trans
        if self.predict_trans:
            self.decode_trans = torch.nn.Sequential(
                torch.nn.Linear(
                    self.bottleneck_size, int(self.bottleneck_size / 2)
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(int(self.bottleneck_size / 2), 3),
            )
        self.predict_scale = predict_scale
        if self.predict_scale:
            self.decode_scale = torch.nn.Sequential(
                torch.nn.Linear(
                    self.bottleneck_size, int(self.bottleneck_size / 2)
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(int(self.bottleneck_size / 2), 1),
            )
            self.decode_scale[-1].bias.data.fill_(1)
        self.predict_draw = predict_draw
        if self.predict_draw:
            self.draw = torch.nn.Sequential(
                    torch.nn.Linear(
                        self.bottleneck_size, int(self.bottleneck_size / 2)
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int(self.bottleneck_size / 2), 3),
            )

        if mode == "sphere" or mode == "attention" or mode == "template":
            test_mesh = trimesh.creation.icosphere(
                subdivisions=inference_ico_divisions
            )

            # Initialize inference vertices and faces
            test_faces = np.array(test_mesh.faces)
            test_verts = test_mesh.vertices
        else:
            raise ValueError("{} not in [sphere]".format(mode))
        self.test_verts = torch.Tensor(
            np.array(test_verts).astype(np.float32)
        ).cuda()
        self.test_faces = test_faces

        self.attentionblock = MultiAttentionLayer(
            n_head=4,
            d_q_=515,
            d_k_=3,
            d_v_=3,
            d_k=256,
            d_v=128,
            d_o=128
        )

    def forward(self, img_features):
        # Predict translation if needed
        if self.predict_trans:
            translations = self.decode_trans(img_features)
        # Sample random points on unit sphere
        rand_grid = img_features.new_empty(
            (img_features.size(0), 3, self.points_nb)
        )
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(
            torch.sum(rand_grid ** 2, dim=1, keepdim=True)
        )

        # Add grid to image features
        img_features = img_features.unsqueeze(2).repeat(
            1, 1, rand_grid.size(2)
        )
        dec_img_features = torch.cat((rand_grid, img_features), 1)

        # Decoder pass
        verts = self.decoder(dec_img_features).transpose(2, 1)
        if self.predict_trans:
            objpoints3d = verts + translations.unsqueeze(1)
            results = {
                "objpoints3d": objpoints3d,
                "objtrans": translations,
                "objpointscentered3d": verts,
            }
        else:
            results = {"objpoints3d": verts}
        return results

    def forward_inference(self, img_features, separate_encoder_features=None):
        # Predict translation if needed
        if self.predict_trans:
            translations = self.decode_trans(img_features)
        if self.predict_scale:
            scales = self.decode_scale(img_features)
        # Initialize test grid
        test_grid = (
            self.test_verts.unsqueeze(0)
            .repeat(img_features.shape[0], 1, 1)
            .transpose(2, 1)
        ).cuda()
        if self.separate_encoder:
            dec_img_features = separate_encoder_features.unsqueeze(2).repeat(
                1, 1, test_grid.size(2)
            )
        else:
            dec_img_features = img_features.unsqueeze(2).repeat(
                1, 1, test_grid.size(2)
            )

        dec_img_features_grid = torch.cat((test_grid, dec_img_features), 1)
        verts = self.decoder(dec_img_features_grid).transpose(2, 1)
        if self.predict_scale:
            scaled_verts = scales.unsqueeze(1) * verts
            if self.predict_trans:
                objpoints3d = scaled_verts + translations.unsqueeze(1)
        elif self.predict_trans:
            objpoints3d = verts + translations.unsqueeze(1)
        if not self.predict_scale and not self.predict_trans:
            results = {"objpoints3d": verts, "objfaces": self.test_faces}
        if self.predict_trans:
            results = {
                "objpoints3d": objpoints3d,
                "objtrans": translations,
                "objpointscentered3d": verts,
                "objfaces": self.test_faces,
            }
        if self.predict_scale:
            results["objscale"] = scales
        return results

    def forward_template(self, img_features, point_feature, template_verts):
        if self.predict_trans:
            translations = self.decode_trans(img_features)
        if self.predict_scale:
            scales = self.decode_scale(img_features)
        elif self.predict_draw:
            draw = self.draw(img_features)

        dec_img_features = img_features.unsqueeze(2).repeat(
            1, 1, 642
        )
        dec_img_features = dec_img_features.permute(0, 2, 1)
        nor_dec_img_features = torch.norm(dec_img_features, p=2, dim=2)
        dec_img_features = (dec_img_features / nor_dec_img_features.unsqueeze(2))

        # Initialize test grid
        template_verts_f = torch.cat([template_verts, dec_img_features], dim=2)
        test_grid = (
            template_verts_f.transpose(2, 1)
        ).cuda()

        delta = self.decoder(test_grid).transpose(2, 1)
        # delta
        verts = template_verts + delta # [bs, 642, 3]
        # atlasnet
        verts = delta # [bs, 642, 3]

        if self.predict_scale:
            scaled_verts = scales.unsqueeze(1) * verts
            if self.predict_trans:
                verts = scaled_verts + translations.unsqueeze(1)
        elif self.predict_draw:
            scaled_verts = verts * draw.unsqueeze(1)
            verts = scaled_verts + translations.unsqueeze(1)
        elif self.predict_trans:
            verts = verts + translations.unsqueeze(1)

        if not self.predict_scale and not self.predict_trans:
            results = {
                "objpoints3d": verts, 
                "objfaces": self.test_faces,
                "objscale": None,
                "objtrans": None,
            }
        if self.predict_trans:
            results = {
                "objpoints3d": verts,
                "objtrans": translations,
                "objfaces": self.test_faces,
                "objscale": None
            }
        if self.predict_scale:
            results["objscale"] = scales
        return results, delta

def edge_loss(edges, faces):
    edges_A = edges[:, faces[:, 0]]
    edges_B = edges[:, faces[:, 1]]
    edges_C = edges[:, faces[:, 2]]
    edge_lengths_A = torch.sum((edges_B - edges_A) ** 2, dim=2)
    edge_lengths_B = torch.sum((edges_C - edges_B) ** 2, dim=2)
    edge_lengths_C = torch.sum((edges_A - edges_C) ** 2, dim=2)
    all_edges = torch.cat(
        [edge_lengths_C, edge_lengths_B, edge_lengths_A], dim=1
    )
    mean_edge_size = all_edges.mean(1, keepdim=True).repeat(
        1, all_edges.shape[1]
    )
    edge_loss = torch.mean(torch.abs(all_edges - mean_edge_size))
    return edge_loss


class AtlasLoss:
    def __init__(
        self,
        lambda_atlas=1,
        atlas_loss="chamfer",
        final_lambda_atlas=1,
        center_weight=0,
        trans_weight=0,
        scale_weight=0,
        reg_weight=0,
        rot_weight=0,
        edge_regul_lambda=None,
        lambda_laplacian=0,
        laplacian_faces=None,
        laplacian_verts=None,
    ):
        self.lambda_atlas = lambda_atlas
        self.final_lambda_atlas = final_lambda_atlas
        self.trans_weight = trans_weight
        self.scale_weight = scale_weight
        self.center_weight = center_weight
        self.reg_weight = reg_weight
        self.rot_weight = rot_weight
        self.edge_regul_lambda = edge_regul_lambda
        self.lambda_laplacian = lambda_laplacian

        self.rotloss = torch.nn.L1Loss()
        if lambda_laplacian:
            self.laplacian_loss = LaplacianLoss(
                laplacian_faces, laplacian_verts
            )
        self.atlas_loss = atlas_loss
        if self.atlas_loss == "chamfer":
            self.chamfer_loss = atlasutils.ChamferLoss()
        else:
            raise ValueError("Removed support for earth mover distance !")

    def compute_loss(self, preds, target):
        atlas_losses = {}
        if (
            TransQueries.objpoints3d in target
            and (self.lambda_atlas or self.final_lambda_atlas)
        ) or (TransQueries.center3d in target and self.trans_weight):
            # Translation + centered object is predicted
            if (
                "objtrans" in preds
                and TransQueries.objpoints3d in target
                and ("objpointscentered3d" in preds)
            ):
                obj_centroids = target[TransQueries.objpoints3d].mean(1)
                trans3d_loss = torch_f.mse_loss(
                    preds["objtrans"], obj_centroids
                )
                atlas_losses["atlas_trans3d"] = trans3d_loss
                centered_objpoints3d = target[
                    TransQueries.objpoints3d
                ] - obj_centroids.unsqueeze(1)
                if "objscale" in preds:
                    obj_scales = torch.norm(centered_objpoints3d, 2, 2).max(1)[
                        0
                    ]
                    scale3d_loss = torch_f.mse_loss(
                        preds["objscale"], obj_scales.unsqueeze(1)
                    )

                    atlas_losses["atlas_scale3d"] = scale3d_loss
                else:
                    scale3d_loss = 0
                # Output in 'objpoints3d' is uncentered
                if self.atlas_loss == "chamfer":
                    loss_1, loss_2 = self.chamfer_loss(
                        preds["objpointscentered3d"], centered_objpoints3d
                    )
                    sym_loss = torch.mean(loss_1 + loss_2)
                obj_mesh = preds["objpointscentered3d"]

                # Chamfer loss with incorporated trans
                if self.atlas_loss == "chamfer":
                    final_loss_1, final_loss_2 = self.chamfer_loss(
                        preds["objpoints3d"], target[TransQueries.objpoints3d]
                    )
                    sym_final_loss = torch.mean(final_loss_1 + final_loss_2)
                atlas_losses[
                    "final_{}_loss".format(self.atlas_loss)
                ] = sym_final_loss
                final_loss = (
                    self.lambda_atlas * sym_loss
                    + self.final_lambda_atlas * sym_final_loss
                    + self.trans_weight * trans3d_loss
                    + self.scale_weight * scale3d_loss
                )
                sym_loss = sym_loss

            else:
                # Translation is not predicted separately
                if "objpoints3d" in preds and self.lambda_atlas:
                    if self.atlas_loss == "chamfer":
                        loss_1, loss_2 = self.chamfer_loss(
                            preds["objpoints3d"],
                            target[TransQueries.objpoints3d],
                        )
                        sym_loss = torch.mean((loss_1 + loss_2))
                    final_loss = self.lambda_atlas * sym_loss
                    obj_mesh = preds["objpoints3d"]
            # Regularization losses
            if self.edge_regul_lambda is not None and (
                self.edge_regul_lambda > 0
            ):
                edge_regul_loss = edge_loss(obj_mesh, preds["objfaces"])
                atlas_losses["atlas_edge_regul"] = edge_regul_loss
                final_loss = (
                    final_loss + self.edge_regul_lambda * edge_regul_loss
                )
            if self.lambda_laplacian:
                laplacian_loss = self.laplacian_loss(obj_mesh)
                atlas_losses["atlas_laplac"] = laplacian_loss
                final_loss = (
                    final_loss + self.lambda_laplacian * laplacian_loss
                )
        else:
            sym_loss = None
            final_loss = torch.Tensor([0]).cuda()

        atlas_losses["atlas_objpoints3d"] = sym_loss

        return final_loss, atlas_losses

    def my_compute_loss(self, pred_verts, pred_trans, pred_scale, pred_rot, gt_rot, target, mask_points=None, delta=None): # target is objectverts_3d
        atlas_losses = {}
        # Translation + centered object is predicted
        obj_centroids = target.mean(1)
        if pred_trans is None:
            pred_trans = pred_verts.mean(1)
        trans3d_loss = torch_f.mse_loss(
            pred_trans, obj_centroids
        )
        atlas_losses["atlas_trans3d"] = trans3d_loss
        centered_objpoints3d = target - obj_centroids.unsqueeze(1)
        centered_pred = pred_verts - pred_trans.unsqueeze(1)

        obj_scales = torch.norm(centered_objpoints3d, 2, 2).max(1)[
            0
        ]
        pred_obj_scale = torch.norm(centered_pred, 2, 2).max(1)[
            0
        ]
        scale3d_loss = torch_f.mse_loss(
            pred_obj_scale, obj_scales
        )

        if delta is not None:
            reg_loss = torch_f.mse_loss(
                delta, torch.zeros_like(delta)
            )
        else:
            reg_loss = 0

        atlas_losses["atlas_scale3d"] = scale3d_loss

        if pred_rot is not None:
            rot_loss = torch_f.mse_loss(pred_rot, gt_rot) * 1000
        else:
            rot_loss = 0

        # Output in 'objpoints3d' is uncentered
        if mask_points is not None:
            loss_1, loss_2 = self.chamfer_loss.forward_mask(
                pred_verts, target, mask_points
            )
        else:
            loss_1, loss_2 = self.chamfer_loss(
                pred_verts, target
            )
        sym_loss = torch.mean(loss_1 + loss_2)

        centered_loss1, centered_loss2 = self.chamfer_loss.calc_dcd(
                centered_pred, centered_objpoints3d
            )
        centered_loss = torch.mean(centered_loss1 + centered_loss2) * 1000

        final_loss = (
            self.final_lambda_atlas * sym_loss
            + self.trans_weight * trans3d_loss
            + self.scale_weight * scale3d_loss
            + self.center_weight * centered_loss
            + self.reg_weight * reg_loss
            + self.rot_weight * rot_loss
        )

        return final_loss, trans3d_loss, scale3d_loss, reg_loss, rot_loss