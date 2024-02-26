import torch
from torch import nn
from copy import deepcopy

from obman_net.mano_train.networks.bases import resnet
from obman_net.mano_train.networks.branches.manobranch import ManoBranch
from obman_net.mano_train.networks.branches.atlasbranch import AtlasBranch

class Obman(nn.Module):
    def __init__(
        self,
        backbone,
    ):
        super(Obman, self).__init__()
        # visual block
        if backbone == "resnet18":
            img_feature_size = 512
            base_net = resnet.resnet18(pretrained=True)            
        elif backbone == "resnet50":
            img_feature_size = 2048
            base_net = resnet.resnet50(pretrained=True)
        self.base_net = base_net
        self.atlas_base_net = deepcopy(base_net)
        # hand branch
        mano_base_neurons = [img_feature_size] + [1024, 256] #hidden_neurons
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
        # atlas branch
        self.atlas_mesh = True
        feature_size = img_feature_size
        self.atlas_branch = AtlasBranch(
            mode="sphere",
            use_residual=False,
            use_unet=False,
            points_nb=600,
            predict_trans=True,
            predict_scale=True,
            inference_ico_divisions=3,
            bottleneck_size=feature_size,
            use_tanh=False,
            out_factor=100,
            separate_encoder=True,
        )
        self.test_faces = self.atlas_branch.test_faces
    
    def forward(
        self, image_crop
    ):
        batch_size = image_crop.shape[0]
        image = image_crop.cuda()
        features, _ = self.base_net(image)
        atlas_infeatures, _ = self.atlas_base_net(image)
        # Mano branch
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
        # Atlas branch
        atlas_features = features
        atlas_results = self.atlas_branch.forward_inference(
            atlas_features,
            separate_encoder_features=atlas_infeatures
        )
        obj_verts = atlas_results["objpoints3d"]
        obj_trans = atlas_results["objtrans"]
        obj_scale = atlas_results["objscale"]
        
        return hand_verts, hand_joints, hand_shape, hand_pose, \
                obj_verts, obj_trans, obj_scale