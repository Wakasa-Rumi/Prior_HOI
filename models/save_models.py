import torch
import os

def save_whole_pkl(model, save_path, epoch):
    save_path = save_path + "/models"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + "/epoch{}.pkl".format(epoch)
    torch.save(model, save_path)

def save_part_pkl(model, save_path, epoch):
    save_path = save_path + "/models/stage1"
    if not os.path.exists(save_path):
        os.makedirs(save_path)   

    torch.save(model.module.hand_net, save_path + "hand_net.pkl")
    torch.save(model.module.pc_encoder, save_path + "pc_encoder.pkl")
    torch.save(model.module.atlas_base_net, save_path + "atlas_base_net.pkl")
    torch.save(model.module.atlas_branch, save_path + "atlas_branch.pkl")
    torch.save(model.module.base_net, save_path + "base_net.pkl")

def save_model(model, save_path, epoch, save_mode):
    if save_mode == "whole_pkl":
        save_whole_pkl(model, save_path, epoch)