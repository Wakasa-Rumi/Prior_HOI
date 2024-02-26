import torch
from models.Obman import Obman
from models.PoseBlock import PoseMano, RotationBlock
from models.RefinementBlock import RefineBlock
from models.Template_HOI import Template_HOI
from hand_pipline.YTBHand_network import spiral_tramsform
from obman_net.mano_train.networks import netutils

def get_models(model_name, backbone_name, refine):
    # prepare for YTBHand
    template_fp = "hand_pipline/template/template.ply"
    transform_fp = "hand_pipline/template/transform.pkl"
    seq_length = [27, 27, 27, 27] # the length of neighbours
    dilation = [1, 1, 1, 1] # whether dilate sample
    ds_factors = [2, 2, 2, 2] # downsample factors 2, 2, 2, 2      
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation)
    
    if model_name == "Obman":
        model = Obman(backbone=backbone_name)
        netutils.freeze_batchnorm_stats(model)
        return model, None, None
    elif model_name == "TemplateHOI":
        model = Template_HOI(
            spiral_indices_list=spiral_indices_list,
            up_transform_list=up_transform_list,
            backbone=backbone_name
        )   
        netutils.freeze_batchnorm_stats(model)

        return model, tmp, down_transform_list
    elif model_name == "PoseHOI":
        if refine == False:
            model = PoseMano(backbone=backbone_name)
        else:
            model = RefineBlock(backbone=backbone_name)
        return model, None, None

def load_model(model, load_path, load_mode):
    if load_mode == 'whole_pkl':
        return load_model_pkl(model, load_path)
    elif load_mode == 'pose_block':
        return load_pose_block(model, load_path)

def load_model_pkl(model, load_path):
    model = torch.load(load_path, map_location=torch.device('cpu'))
    return model

def load_pose_block(model, load_path):
    model.pose_block = torch.load(load_path, map_location=torch.device('cpu'))
    # netutils.rec_freeze(model.pose_block.base_net)
    # netutils.rec_freeze(model.pose_block.pc_encoder)
    # netutils.rec_freeze(model.pose_block.rotation_block)
    # netutils.rec_freeze(model.pose_block.trans_block)
    # netutils.rec_freeze(model.pose_block.scale_block)
    netutils.rec_freeze(model.pose_block)
    return model

def load_model_pth(model, save_pth):
    pass