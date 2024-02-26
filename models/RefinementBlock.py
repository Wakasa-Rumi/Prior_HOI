from copy import deepcopy
import pickle
import torch
from torch import nn
import torch.nn.functional as F

from obman_net.mano_train.networks.bases import resnet, pointnet
from obman_net.mano_train.networks.branches.atlasbranch import AtlasBranch
from obman_net.mano_train.networks.branches.manobranch import ManoBranch
from hand_pipline.YTBHand_network import YTBHand
from models.attention_utils.Projecter import project
from models.PoseBlock import PoseMano

class RefineBlock(nn.Module):
    def __init__(
        self,
        atlas_separate_encoder=True,
        backbone=None
    ):
        super(RefineBlock, self).__init__()

        # visual model
        base_net = resnet.resnet18(pretrained=True, return_inter=True)
        img_feature_num = 512

        # atlas net
        self.atlas_separate_encoder = atlas_separate_encoder

        self.base_net = base_net
        self.img_feature_num = img_feature_num
        if self.atlas_separate_encoder:
            self.atlas_base_net = deepcopy(base_net)

        # pose block
        self.pose_block = PoseMano(backbone)
        
        # object branch
        self.atlas_branch = AtlasBranch(
            mode="template",
            use_residual=False,
            use_unet=False,
            points_nb=642,
            predict_trans=True,
            predict_scale=True,
            predict_draw=False,
            inference_ico_divisions=3,
            bottleneck_size=img_feature_num,
            use_tanh=False,
            out_factor=100,
            separate_encoder=True,
        )
        self.test_faces = self.atlas_branch.test_faces

    def interpolate_feature_map(self, feature_maps_dict):
        feature_maps = feature_maps_dict["res_layer2"]
        return feature_maps # [bs, 512, 64, 64]

    def forward(
        self, image_crop, prior, uni_prior, camera
    ):
        image = image_crop.cuda()
        features, _ = self.base_net(image)
        atlas_infeatures, feature_maps_dict = self.atlas_base_net(image)

        (hand_verts, hand_joints, hand_shape, hand_pose,
            rot_matrix, trans, scale) = self.pose_block(image_crop, uni_prior)
        pose_prior = torch.bmm(prior, rot_matrix) * torch.abs(scale).unsqueeze(1) + trans.unsqueeze(1)

        feature_maps = self.interpolate_feature_map(feature_maps_dict).cuda()
        point_feature = project(pose_prior, camera.cuda(), feature_maps)

        # atlas
        atlas_features = features
        atlas_infeatures = atlas_infeatures

        atlas_results, _ = self.atlas_branch.forward_template(
            atlas_features,
            point_feature.cuda(),
            template_verts=pose_prior
        )

        # got all results
        pred_verts = atlas_results["objpoints3d"]

        return pred_verts, hand_verts, hand_joints, hand_shape, hand_pose, \
                trans, scale