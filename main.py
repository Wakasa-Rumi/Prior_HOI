# python repo
import os
import json
import time
import random
import torch
import warnings
import numpy as np
from matplotlib import pyplot as plt

# my repo
from dataloader.datasets import get_dataset
from models.get_models import get_models, load_model
from models.save_models import save_model
from loss.losses import get_loss
from loss.acc import save_pck_file
from running.epoch_dexycb import epoch_dexycb_train, epoch_dexycb_val, epoch_dexycb_test
from running.epoch_ho3d import epoch_ho3d_train, epoch_ho3d_val
from running.epoch_obman import epoch_obman_dexycb_train, epoch_obman_dexycb_eval
from running.epoch_posehoi import epoch_posehoi_ho3d_train, epoch_posehoi_ho3d_eval

args = {
    # running mode
    "name": "Obman_2_26",
    "cuda": "0",
    "batch_size": 32,
    "mode": "train", # train, test
    "epoch": 30,
    "lr": 1e-4,
    "lr_pb": 1e-4,
    "lr_decay_step": 300,
    "lr_decay_gamma": 0.5,
    "display_train_freq": 1000,
    "display_val_freq": 100,
    "display_test_freq": 100,
    # data related
    "dataset": "dexycb", # ho3d, dexycb
    # model related
    "backbone": "resnet18", # resnet18, resnet50
    "model_name": "Obman", # Obman, PoseBlock, TemplateHOI, PoseHOI
    "refine": False,
    "load_mode": "whole_pkl", # dg_pth, whole_pkl, pose_block
    "load_path": "/home/yiyao/HOI/HOI/ho/checkpoints1/train_Obman_resnet18_dexycb/models/epochbest.pkl",
    "save_mode": "whole_pkl",
    "save_pth": "checkpoints3",
    "save_freq": 1,
    "save_begin": 0,
    # output
    "global_loss": 9999999.9,
    "current_loss": 0,
    "current_epoch": 0,
    "best_epoch": 0,
    "start_time": "",
    "finish_time": "",
    "comment": "mask_weight = 2, trans and scale use image feature, rot use both image feature and 3D feature, mesh loss not align, try render loss + mesh loss"
}

args = {
    # running mode
    "name": "Pose_2_26",
    "cuda": "0",
    "batch_size": 32,
    "mode": "train", # train, test
    "epoch": 30,
    "lr": 5e-5,
    "lr_pb": 5e-5,
    "lr_decay_step": 300,
    "lr_decay_gamma": 0.5,
    "display_train_freq": 400,
    "display_val_freq": 100,
    "display_test_freq": 100,
    # data related
    "dataset": "ho3d", # ho3d, dexycb
    # model related
    "backbone": "resnet18", # resnet18, resnet50
    "model_name": "PoseHOI", # Obman, PoseBlock, TemplateHOI, PoseHOI
    "refine": False,
    "load_mode": None, # dg_pth, whole_pkl, pose_block
    "load_path": "/home/yiyao/HOI/checkpoints_new/week5/2024_2_19_priorhoi/models/epochbest.pkl",
    "save_mode": "whole_pkl",
    "save_pth": "checkpoints4",
    "save_freq": 0,
    "save_begin": 0,
    # output
    "global_loss": 9999999.9,
    "current_loss": 0,
    "current_epoch": 0,
    "best_epoch": 0,
    "start_time": "",
    "finish_time": "",
    "comment": "rot, trans, scale decoder, mesh loss center"
}
os.environ['CUDA_VISIBLE_DEVICES']=args["cuda"]

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if not os.path.exists(args["save_pth"]):
        os.mkdir(args["save_pth"])

    device = torch.device("cuda")

    # settings
    start_time = time.localtime()
    time_str = str(start_time.tm_mon) + "/" + str(start_time.tm_mday) \
                + " " + str(start_time.tm_hour) + ":" + str(start_time.tm_min)
    args["start_time"] = time_str

    # set save feature
    fig = plt.figure(figsize=(12, 12))
    save_path = args["save_pth"] + "/" + \
                args["mode"] + "_" + args["model_name"] + "_" + \
                args["backbone"] + "_" + args["dataset"]
    if args["refine"] == True:
        save_path = save_path + "_" + "refine"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args["save_pth"] = save_path

    json_str = json.dumps(args, indent=4)
    with open(save_path+"/start_args.json", "w", newline="\n") as f:
        f.write(json_str)
        print("start!")

    # datasets
    data_train, data_val, data_test = get_dataset(dataset_name=args["dataset"], model_name=args["model_name"])
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=args["batch_size"], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=args["batch_size"], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=args["batch_size"], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # models
    model, hand_tmp, down_transform_list = get_models(model_name=args["model_name"], backbone_name=args["backbone"], refine=args["refine"])
    if args["load_mode"] is not None:
        model = load_model(model, args["load_path"], args["load_mode"])
    # model = torch.nn.DataParallel(model)
    model.cuda()
    # print(model)

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_param_names = [
        name for name, val in model.named_parameters() if val.requires_grad
    ]
    if args["mode"] == "train" and args["model_name"] == "TemplateHOI":
        ignored_params = list(map(id, model.module.pose_block.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        poseblock_params = model.module.pose_block.parameters()
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr':args["lr"]},
            {'params': poseblock_params, 'lr':args["lr_pb"]}
        ], lr=args["lr"], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args["lr_decay_step"], gamma=args["lr_decay_gamma"]
        )
    else:
        optimizer = torch.optim.Adam(model_params, lr=args["lr"], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args["lr_decay_step"], gamma=args["lr_decay_gamma"]
        )

    # loss model
    loss_model = get_loss(args)

    # start training
    if args["mode"] == "train":
        for epoch in range(0, args["epoch"]):
            # train
            if args["dataset"] == "dexycb":
                if args["model_name"] == "Obman":
                    epoch_obman_dexycb_train(
                        train_loader=data_loader_train,
                        model=model,
                        loss_model=loss_model,
                        epoch=epoch,
                        optimizer=optimizer,
                        args=args, 
                        fig=fig
                    )
                    # eval
                    with torch.no_grad():
                        loss_now = epoch_obman_dexycb_eval(
                            eval_loader=data_loader_val,
                            model=model,
                            loss_model=loss_model,
                            epoch=epoch,
                            args=args, 
                            fig=fig
                        )                    
                else:
                    epoch_dexycb_train(
                        train_loader=data_loader_train,
                        model=model,
                        loss_model=loss_model,
                        epoch=epoch,
                        optimizer=optimizer,
                        hand_tmp=hand_tmp, args=args, fig=fig,
                        down_transform_list=down_transform_list
                    )
                    # eval
                    with torch.no_grad():
                        loss_now = epoch_dexycb_val(
                            eval_loader=data_loader_val,
                            model=model,
                            loss_model=loss_model,
                            epoch=epoch,
                            hand_tmp=hand_tmp, args=args, fig=fig,
                            down_transform_list=down_transform_list
                        )
            elif args["dataset"] == "ho3d": 
                if args["model_name"] == "PoseHOI":
                    epoch_posehoi_ho3d_train(
                        train_loader=data_loader_train,
                        model=model,
                        loss_model=loss_model,
                        epoch=epoch,
                        optimizer=optimizer,
                        args=args,
                        fig=fig,
                        device=device
                    )
                    with torch.no_grad():
                        loss_now = epoch_posehoi_ho3d_eval(
                            eval_loader=data_loader_val,
                            model=model,
                            loss_model=loss_model,
                            epoch=epoch,
                            args=args,
                            fig=fig,
                            device=device
                        )  
            scheduler.step()
            # save model
            if epoch >= args["save_begin"] and loss_now <= args["global_loss"]:
                args["best_epoch"] = epoch
                args["global_loss"] = loss_now
                save_model(model, save_path, "best", "whole_pkl")
                print("Update: global_cd = ", args["global_loss"], " best_epoch =", args["best_epoch"])
            elif epoch % args["save_freq"] == 0:
                save_model(model, save_path, epoch, "whole_pkl")
                print("No_Update: global_cd =", args["global_loss"], " best_epoch = ", args["best_epoch"])
                print("Save: epoch = ", epoch)
            else:
                print("No_Update: global_cd =", args["global_loss"], " best_epoch = ", args["best_epoch"])
                print("epoch = ", epoch)

            args["current_loss"] = loss_now
            args["current_epoch"] = epoch
            json_str = json.dumps(args, indent=4)
            with open(save_path+"/current_args.json", "w", newline="\n") as f:
                f.write(json_str)
                print("done!")

    elif args["mode"] == "test":
        with torch.no_grad():
            if args["dataset"] == "dexycb":
                if args["model_name"] == "Obman":
                    pck_info_hand, pck_info_10, pck_info_5 = epoch_dexycb_test(
                        test_loader=data_loader_test,
                        model=model,
                        loss_model=loss_model,
                        epoch=0,
                        hand_tmp=hand_tmp,
                        args=args, fig=fig,
                        down_transform_list=down_transform_list
                    )
                    save_pck_file(pck_info_hand, pck_info_5, pck_info_10, args)      
            elif args["dataset"] == "ho3d":
                loss_now, pck_info_10, pck_info_5 = epoch_ho3d_val(
                    test_loader=data_loader_val,
                    model=model,
                    loss_model=loss_model,
                    epoch=0,
                    args=args,
                    fig=fig
                )        
                save_pck_file(None, pck_info_5, pck_info_10, args)
    
    # finish training
    finish_time = time.localtime()
    time_str = str(finish_time.tm_mon) + "/" + str(finish_time.tm_mday) \
                + " " + str(finish_time.tm_hour) + ":" + str(finish_time.tm_min)
    args["finish_time"] = time_str

    json_str = json.dumps(args, indent=4)
    with open(save_path+"/finish_args.json", "w", newline="\n") as f:
        f.write(json_str)
        print("done!")