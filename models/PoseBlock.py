import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import deepcopy
from obman_net.mano_train.networks.bases import resnet, pointnet
from models.pose_utils.compute_pose import sixpose2rotmatrix
from obman_net.mano_train.networks.branches.manobranch import ManoBranch

class PoseMano(nn.Module):
    def __init__(
        self,
        backbone,
    ):
        super(PoseMano, self).__init__()
        # visual model
        if backbone == "resnet18":
            base_net = resnet.resnet18(pretrained=True)
            img_feature_num = 512
            feature_dim = img_feature_num
        self.hand_base_net = deepcopy(base_net)
        self.base_net = base_net
        self.img_feature_num = img_feature_num
        self.feature_dim = feature_dim

        self.pc_encoder = pointnet.PointNet_MSG()

        mano_base_neurons = [img_feature_num] + [1024, 256] #hidden_neurons
        self.absolute_lambda = 0
        self.mano_branch = ManoBranch(
            ncomps=30,
            base_neurons=mano_base_neurons,
            adapt_skeleton=False,
            dropout=0,
            use_trans=False,
            mano_root="misc/mano",
            center_idx=0,
            use_shape=True,
            use_pca=True,
        )

        self.rotation_block = nn.Sequential(
            nn.Linear(
                self.feature_dim + 1024, int(self.feature_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 6),
            nn.Dropout(0.5)
        )
        self.trans_block = nn.Sequential(
            nn.Linear(
                self.feature_dim, int(self.feature_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 3),
            nn.Dropout(0.5)
        )
        self.scale_block = nn.Sequential(
            nn.Linear(
                self.feature_dim, int(self.feature_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 1),
            nn.Dropout(0.5)
        )

        # self.pose_block = nn.Sequential(
        #     nn.Linear(
        #         self.feature_dim, int(self.feature_dim / 2)
        #     ),
        #     nn.ReLU(),
        #     nn.Linear(int(self.feature_dim / 2), 10),
        #     nn.Dropout(0.5)
        # )
    
    def forward(self, image_crop, obj_coarse_pc):
        image = image_crop.cuda()
        batch_size = image_crop.shape[0]

        # predict hand
        hand_features, _ = self.hand_base_net(image)
        sides = ["right"] * batch_size
        mano_results = self.mano_branch(
            hand_features,
            sides=sides,
            root_palm=False,
            use_stereoshape=False,
        )
        hand_verts = mano_results["verts"]
        hand_joints = mano_results["joints"]
        hand_shape = mano_results["shape"]
        hand_pose = mano_results["pose"]

        # predict object
        img_features, _ = self.base_net(image) # 2D feature # [bs, 1024]
        _, _, pc_features = self.pc_encoder(obj_coarse_pc) # 3D feature

        # img_features = img_features / torch.norm(img_features, dim=1).unsqueeze(-1)
        # pc_features = pc_features / torch.norm(pc_features, dim=1).unsqueeze(-1)

        features = torch.cat([img_features, pc_features], dim=1)
        rot = self.rotation_block(features)
        rot_matrx = sixpose2rotmatrix(rot)

        trans = self.trans_block(img_features) # [bs, 3]
        scale = self.scale_block(img_features)  # [bs, 1]

        # ten_pose = self.pose_block(img_features)
        # scale = ten_pose[:, :1]
        # trans = ten_pose[:, 1:4]
        # rot = ten_pose[:, 4:]
        # rot_matrx = sixpose2rotmatrix(rot)

        return hand_verts, hand_joints, hand_shape, hand_pose, \
               rot_matrx, trans, scale

class RotationBlock(nn.Module):
    def __init__(
        self,
        backbone,
    ):
        super(RotationBlock, self).__init__()
        # visual model
        if backbone == "resnet18":
            base_net = resnet.resnet18(pretrained=True)
            img_feature_num = 512
            feature_dim = 512
        elif backbone == "resnet50":
            base_net = resnet.resnet50(pretrained=True)
            img_feature_num = 2048
            feature_dim = 2048
        self.base_net = base_net
        self.img_feature_num = img_feature_num
        self.feature_dim = feature_dim

        self.pc_encoder = pointnet.PointNet_MSG()

        self.rotblock = nn.Sequential(
            nn.Linear(
                self.feature_dim, int(self.feature_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 6),
            nn.Dropout(0.5)
        )

        self.auxiliary = nn.Sequential(
            nn.Linear(
                self.feature_dim, int(self.feature_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 6),
            nn.Dropout(0.5)
        )
    
    def forward(self, image_crop, obj_coarse_pc):
        # obj_coarse_pc[bs, N, 3]
        image = image_crop.cuda()
        features, _ = self.base_net(image)
        # _, _, pc_features = self.pc_encoder(obj_coarse_pc)

        # features = torch.cat([features, pc_features], dim=1)
        six_pose = self.rotblock(features)

        rot_matrx = sixpose2rotmatrix(six_pose)

        # trans = self.transblock(features)

        prior = torch.bmm(obj_coarse_pc, rot_matrx)

        return prior, rot_matrx
    
    def forward_auxiliary(self, prior_mask, gt_mask, prior, gt):
        prior_mask = prior_mask.cuda()
        gt_mask = gt_mask.cuda()

        prior_features, _ = self.base_net(prior_mask)
        gt_features, _ = self.base_net(gt_mask)

        prior_six_pose = self.rotblock(prior_features)
        gt_six_pose = self.rotblock(gt_features)

        prior_rot_matrix = sixpose2rotmatrix(prior_six_pose)
        gt_rot_matrix = sixpose2rotmatrix(gt_six_pose)

        rot_prior = torch.bmm(prior, prior_rot_matrix)
        rot_gt = torch.bmm(gt, gt_rot_matrix)

        return rot_prior, rot_gt, gt_rot_matrix