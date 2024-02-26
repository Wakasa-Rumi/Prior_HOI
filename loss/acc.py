import os
import torch
import pickle

from obman_net.mano_train.visualize import displaymano
from obman_net.mano_train.networks.branches import atlasutils

def compute_chamfer(obj_point_input, pred_verts, shape_prior=None):
    c_loss = atlasutils.ChamferLoss()

    gt_verts_center = obj_point_input.mean(dim=1).unsqueeze(1)
    pred_verts_center = pred_verts.mean(dim=1).unsqueeze(1)

    c_gt = obj_point_input - gt_verts_center
    c_pr = pred_verts - pred_verts_center

    if shape_prior is not None:
        shape_prior_center = shape_prior.mean(dim = 1).unsqueeze(1)
        c_sp = shape_prior - shape_prior_center
        pr1, pr2 = c_loss.cdc_p(c_sp, c_gt)
        chamfer_distance_pr = pr1 + pr2
        chamfer_distance_pr = torch.mean(chamfer_distance_pr, dim=-1).item()
    else:
        chamfer_distance_pr = None
        c_sp = None

    d1, d2 = c_loss.cdc_p(obj_point_input, pred_verts)
    chamfer_distance = d1 + d2

    d1, d2 = c_loss.cdc_p(c_gt, c_pr)
    chamfer_distance_center = d1 + d2

    chamfer_distance = torch.mean(chamfer_distance, dim=-1).item()
    chamfer_distance_center = torch.mean(chamfer_distance_center, dim=-1).item()
    return c_sp, chamfer_distance_pr, chamfer_distance, chamfer_distance_center

def test_hand_acc(evaluator):
    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 0.2, 20)

    pck_info_hand = {
        "auc": auc_all,
        "thres": thresholds,
        "pck_curve": pck_curve_all,
        "epe_mean": epe_mean_all,
        "epe_median": epe_median_all,
        "evaluator": evaluator,
    }
    
    return pck_info_hand

def test_obj_acc(evaluator):
    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 10, 20)
    pck_info_10 = {
        "auc": auc_all,
        "thres": thresholds,
        "pck_curve": pck_curve_all,
        "epe_mean": epe_mean_all,
        "epe_median": epe_median_all,
        "evaluator": evaluator,
    }


    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 5, 20)
    pck_info_5 = {
        "auc": auc_all,
        "thres": thresholds,
        "pck_curve": pck_curve_all,
        "epe_mean": epe_mean_all,
        "epe_median": epe_median_all,
        "evaluator": evaluator,
    }
    return pck_info_10, pck_info_5

def save_pck_file(pck_info_hand, pck_info_5, pck_info_10, args):
    pck_folder = os.path.join(args["save_pth"], "pcks")
    os.makedirs(pck_folder, exist_ok=True)

    # save hand
    if pck_info_hand is not None:
        save_pck_file = os.path.join(pck_folder, "hand_eval.png")
        displaymano.save_pck_img(
            pck_info_hand["thres"], 
            pck_info_hand["pck_curve"],
            pck_info_hand["auc"], 
            save_pck_file, overlay=None
        )
        save_pck_pkl = os.path.join(pck_folder, "hand_eval.pkl")
        with open(save_pck_pkl, "wb") as p_f:
            pickle.dump(pck_info_hand, p_f)
    
    # save obj5
    save_pck_file = os.path.join(pck_folder, "obj_eval_5.png")
    displaymano.save_pck_img(
        pck_info_5["thres"], 
        pck_info_5["pck_curve"],
        pck_info_5["auc"], 
        save_pck_file, overlay=None
    )
    save_pck_pkl = os.path.join(pck_folder, "obj_eval_5.pkl")
    with open(save_pck_pkl, "wb") as p_f:
        pickle.dump(pck_info_5, p_f)
    
    # save obj10
    save_pck_file = os.path.join(pck_folder, "obj_eval_10.png")
    displaymano.save_pck_img(
        pck_info_10["thres"], 
        pck_info_10["pck_curve"],
        pck_info_10["auc"], 
        save_pck_file, overlay=None
    )
    save_pck_pkl = os.path.join(pck_folder, "obj_eval_10.pkl")
    with open(save_pck_pkl, "wb") as p_f:
        pickle.dump(pck_info_10, p_f)