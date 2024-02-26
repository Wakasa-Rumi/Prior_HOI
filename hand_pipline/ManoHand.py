import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

import sys
sys.path.append('/home/yiyao/HOI/source_codes/manopth-master')
from manopth.manolayer import ManoLayer

class ManoBranch(nn.Module):
    def __init__(
        self,
        ncomps=6,
        base_neurons=[1024, 512],
        center_idx=9,
        use_shape=False,
        use_trans=False,
        use_pca=True,
        mano_root="misc/mano",
        dropout=0,
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super(ManoBranch, self).__init__()

        self.use_trans = use_trans
        self.use_shape = use_shape
        self.use_pca = use_pca
        self.stereo_shape = torch.Tensor(
            [
                -0.00298099,
                -0.0013994,
                -0.00840144,
                0.00362311,
                0.00248761,
                0.00044125,
                0.00381337,
                -0.00183374,
                -0.00149655,
                0.00137479,
            ]
        ).cuda()

        mano_pose_size = ncomps + 3

        # Base layers
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            zip(base_neurons[:-1], base_neurons[1:])
        ):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            # Initialize all nondiagonal items on rotation matrix weights to 0
            self.pose_reg.bias.data.fill_(0)
            weight_mask = (
                self.pose_reg.weight.data.new(np.identity(3))
                .view(9)
                .repeat(16)
            )
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, 256).float()
                * self.pose_reg.weight.data
            )

        # Shape layers
        self.shape_reg = torch.nn.Sequential(
            nn.Linear(base_neurons[-1], 10)
        )

        # Trans layers(use_trans = False)
        # self.trans_reg = nn.Linear(base_neurons[-1], 3)

        # Mano layers
        self.mano_layer_right = ManoLayer(
            mano_root=mano_root
        )

        self.faces = self.mano_layer_right.th_faces

    def forward(
        self,
        inp, # features
        root_palm=False,
        shape=None,
        pose=None,
    ):
        base_features = self.base_layer(inp)
        pose = self.pose_reg(base_features)
        mano_pose = pose

        # Get trans
        trans = torch.zeros(1)

        # Get shape
        shape = self.shape_reg(base_features)

        # Get pose
        pose_right = mano_pose

        # Pass through mano_right and mano_left layers
        verts_right, joints_right = self.mano_layer_right(
            th_pose_coeffs=pose_right,
            th_betas=shape,
            # th_trans=trans,
            # root_palm=root_palm,
        )

        return verts_right, joints_right, shape, pose


class ManoLoss():
    def __init__(
        self
    ):
        self.lambda_verts = 0.167
        self.lambda_joints3d = 0.167
        self.lambda_shape = 0.167
        self.lambda_pose_reg = 0.167
        self.lambda_pca = 0
        self.center_idx = 9

    def compute_loss(self, preds_verts, target_verts, 
                     pred_joints, target_joints,
                     pred_shape, pred_pose):
        final_loss = 0

        # If needed, compute and add vertex loss
        verts3d_loss = torch_f.mse_loss(
            preds_verts, target_verts
        )
        final_loss += self.lambda_verts * verts3d_loss
        verts3d_loss = verts3d_loss

        # Compute joints loss in all cases
        joints3d_loss = torch_f.mse_loss(pred_joints, target_joints)
        final_loss += self.lambda_joints3d * joints3d_loss

        shape_loss = torch_f.mse_loss(
            pred_shape, torch.zeros_like(pred_shape)
        )
        final_loss += self.lambda_shape * shape_loss
        shape_loss = shape_loss

        pose_reg_loss = torch_f.mse_loss(
            pred_pose[:, 3:], torch.zeros_like(pred_pose[:, 3:])
        )
        final_loss += self.lambda_pose_reg * pose_reg_loss

        return final_loss
