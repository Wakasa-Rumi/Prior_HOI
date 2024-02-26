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
from models.PoseBlock import RotationBlock

class Template_HOI(nn.Module):
    def __init__(
        self,
        atlas_separate_encoder=True,
        spiral_indices_list=None,
        up_transform_list=None,
        backbone=None
    ):
        super(Template_HOI, self).__init__()

        # visual model
        if backbone == "resnet18":
            base_net = resnet.resnet18(pretrained=True, return_inter=True)
            img_feature_num = 512
            feature_dim = 3084
        elif backbone == "resnet50":
            base_net = resnet.resnet50(pretrained=True)
            img_feature_num = 2048
            feature_dim = 5120

        # feature maps
        dim = 64+128+256+512
        out_dim = 512
        self.z_head = nn.Conv2d(dim, out_dim, 1)

        self.atlas_separate_encoder = atlas_separate_encoder

        self.base_net = base_net
        self.img_feature_num = img_feature_num
        if self.atlas_separate_encoder:
            self.atlas_base_net = deepcopy(base_net)

        # hand branch
        # with open("obman_net/misc/mano/MANO_RIGHT.pkl", 'rb') as f:
        #     self.mano_data = pickle.load(f, encoding='latin1')

        # self.hand_net = YTBHand(spiral_indices_list, up_transform_list)
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

        self.pose_block = RotationBlock(backbone)

    def xyz_from_vertice(self, vertice):
        self.Jreg = self.mano_data['J_regressor']
        np_J_regressor = torch.from_numpy(self.Jreg.toarray().T).float()

        joint_x = torch.matmul(vertice[:, :, 0], np_J_regressor.cuda())
        joint_y = torch.matmul(vertice[:, :, 1], np_J_regressor.cuda())
        joint_z = torch.matmul(vertice[:, :, 2], np_J_regressor.cuda())
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
        coords_kp_xyz3 = self.get_keypoints_from_mesh_np(vertice, joints)

        return coords_kp_xyz3

    def get_keypoints_from_mesh_np(self, mesh_vertices, keypoints_regressed):
        """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
        kpId2vertices = {
            4: [744],  # ThumbT
            8: [320],  # IndexT
            12: [443],  # MiddleT
            16: [555],  # RingT
            20: [672]  # PinkT
        }
        keypoints = [0.0 for _ in range(21)]  # init empty list

        # fill keypoints which are regressed
        mapping = {0: 0,  # Wrist
                   1: 5, 2: 6, 3: 7,  # Index
                   4: 9, 5: 10, 6: 11,  # Middle
                   7: 17, 8: 18, 9: 19,  # Pinky
                   10: 13, 11: 14, 12: 15,  # Ring
                   13: 1, 14: 2, 15: 3}  # Thumb

        for manoId, myId in mapping.items():
            keypoints[myId] = keypoints_regressed[:,manoId, :]

        # get other keypoints from mesh
        for myId, meshId in kpId2vertices.items():
            keypoints[myId] = torch.mean(mesh_vertices[:,meshId, :], 1)

        keypoints = torch.stack(keypoints)
        return keypoints  

    def interpolate_feature_map(self, feature_maps_dict):
        feature_maps = feature_maps_dict["res_layer2"]
        return feature_maps # [bs, 512, 64, 64]

    def forward(
        self, image_crop, mask_crop, mix_mask_crop, obj_coarse_pc, camera, center_2d, scale_2d, epoch
    ):
        batch_size = image_crop.shape[0]
        image = image_crop.cuda()
        features, _ = self.base_net(image)
        atlas_infeatures, feature_maps_dict = self.atlas_base_net(image)

        # mask_crop = mask_crop.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        rot_obj_coarse_pc, _ = self.pose_block(mix_mask_crop, obj_coarse_pc)
        # mask_crop = mask_crop.permute(0, 2, 3, 1)[:,:,:,0].unsqueeze(-1).permute(0, 3, 1, 2)

        feature_maps = self.interpolate_feature_map(feature_maps_dict)
        point_feature = project(rot_obj_coarse_pc, camera, feature_maps)

        # hand_net
        # hand_verts_list = self.hand_net(image)
        # hand_joint = self.xyz_from_vertice(hand_verts_list[-1]).permute(1, 0, 2)
        sides = ["right"] * batch_size
        mano_results = self.mano_branch(
            features,
            sides=sides,
            root_palm=False,
            use_stereoshape=False,
        )
        hand_verts = mano_results["verts"]
        hand_joints = mano_results["joints"]
        hand_shape = mano_results["shape"]
        hand_pose = mano_results["pose"]


        # atlas
        atlas_features = features
        atlas_infeatures = atlas_infeatures

        atlas_results, delta = self.atlas_branch.forward_template(
            atlas_features,
            point_feature.cuda(),
            template_verts=rot_obj_coarse_pc,
            epoch=epoch
        )

        # got all results
        pred_verts = atlas_results["objpoints3d"]
        pred_trans = atlas_results["objtrans"]
        pred_scale = atlas_results["objscale"]

        mask_point = project(pred_verts, camera, mask_crop)

        # return pred_verts, pred_trans, pred_scale, \
        #     hand_verts_list, hand_joint, \
        #     mask_point, rot_obj_coarse_pc, delta

        return pred_verts, pred_trans, pred_scale, \
            hand_verts, hand_joints, hand_shape, hand_pose, \
            mask_point, rot_obj_coarse_pc, delta