import os
import torch
import pickle
from progress.bar import Bar as Bar
from obman_net.mano_train.evaluation.evalutils import AverageMeters
from obman_net.mano_train.visualize import displaymano
from loss.acc import compute_chamfer
from loss.Log import write_log
import math

def epoch_obman_dexycb_train(
    train_loader,
    model,
    loss_model,
    epoch,
    optimizer,
    args,
    fig
):
    avg_meters = AverageMeters()
    with open("obman_net/misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        hand_faces = mano_right_data["f"]
    atlas_mesh = model.test_faces

    model.eval()
    save_img_folder = os.path.join(
        args["save_pth"], "images", "train", "epoch_{}".format(epoch)
    )
    save_log_folder = os.path.join(
        args["save_pth"], "logs", "train", "epoch_{}".format(epoch)
    )
    os.makedirs(save_img_folder, exist_ok=True)
    os.makedirs(save_log_folder, exist_ok=True)

    bar = Bar("Processing", max=len(train_loader))
    chamfer_distances = []
    chamfer_center_distances = []

    for step, (image_crop, obj_verts_gt, handverts_gt, handjoint_gt) in enumerate(train_loader):
        obj_verts_gt = obj_verts_gt.float().cuda()
        handverts_gt = handverts_gt.float().cuda()
        handjoint_gt = handjoint_gt.float().cuda()
        # predict
        (
            pred_handverts, pred_handjoint, pred_shape, pred_pose,
            obj_verts, trans, scale
        ) = model(image_crop)
        # loss
        (
            obj_loss, hand_loss, contact_loss, total_loss
        ) = loss_model.compute_loss(
            obj_verts, trans, scale, obj_verts_gt,
            pred_handverts, pred_handjoint, pred_shape, pred_pose,
            handverts_gt, handjoint_gt,
            hand_faces, atlas_mesh
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model_losses = {}
        model_losses["total_loss"] = total_loss
        model_losses["object_loss"] = obj_loss
        model_losses["hand_loss"] = hand_loss
        model_losses["contact_loss"] = torch.tensor(contact_loss)
        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)
        # eval
        _, _, cd, cdc = compute_chamfer(obj_verts_gt, obj_verts)
        if not math.isnan(cd):
            chamfer_distances.append(cd)
        if not math.isnan(cdc):
            chamfer_center_distances.append(cdc)
        # display
        save_img_path = os.path.join(
            save_img_folder, "img_step{}.png".format(step)
        )      
        save_log_path = os.path.join(
            save_log_folder, "log_step{}.json".format(step)
        )  
        if args["display_train_freq"] != 0 and step % args["display_train_freq"] == 0:
            image_crop = image_crop.permute(0, 2, 3, 1)
            # mask_crop = torch.cat([mask_gt.unsqueeze(-1), pred_mask.unsqueeze(-1).detach().cpu(), blue_channel.unsqueeze(-1)], dim=-1) #[128, 128, 3]
            displaymano.visualize_batch(
                args,
                "train",
                save_img_path,
                fig = fig,
                images = image_crop,
                # mask_crop=mask_crop,
                # obj
                objpoints3d_gt = obj_verts_gt,
                preds_objverts = obj_verts,
                obj_coarse_pc = obj_verts,
                # mask_pc=mask_points,
                # hand
                preds_handjoints = pred_handjoint,
                preds_handverts = pred_handverts,
                handjoints_3d_gt = handjoint_gt,
                # faces
                preds_objface = atlas_mesh,
                hand_faces=hand_faces,
                # camera=camera
            )  
            write_log(
                mesh_loss=avg_meters.average_meters["total_loss"].avg,
                scale_loss=avg_meters.average_meters["object_loss"].avg,
                hand_loss=avg_meters.average_meters["hand_loss"].avg,
                contact_loss=avg_meters.average_meters["contact_loss"].avg,
                CDC=sum(chamfer_distances) / len(chamfer_distances),
                CD=sum(chamfer_center_distances) / len(chamfer_center_distances),
                path=save_log_path
            )
        # bar
        bar.suffix = "({batch}/{size}) Total: {total_loss:.2f} | Param: {obj_loss:.2f} | Hand: {hand_loss:.2f} | Contact: {contact_loss:.2f} | CDC: {chamfer_distance_center:.2f} | CD: {chamfer_distance:.2f}".format(
            batch=step + 1,
            size=len(train_loader),
            total_loss = avg_meters.average_meters["total_loss"].avg,
            obj_loss = avg_meters.average_meters["object_loss"].avg,
            hand_loss = avg_meters.average_meters["hand_loss"].avg,
            contact_loss = avg_meters.average_meters["contact_loss"].avg,
            chamfer_distance=sum(chamfer_distances) / len(chamfer_distances),
            chamfer_distance_center=sum(chamfer_center_distances) / len(chamfer_center_distances)
        )
        bar.next()  
    bar.finish()

def epoch_obman_dexycb_eval(
    eval_loader,
    model,
    loss_model,
    epoch,
    args,
    fig
):
    avg_meters = AverageMeters()
    with open("obman_net/misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        hand_faces = mano_right_data["f"]
    atlas_mesh = model.test_faces

    model.eval()
    save_img_folder = os.path.join(
        args["save_pth"], "images", "test", "epoch_{}".format(epoch)
    )
    os.makedirs(save_img_folder, exist_ok=True)
    save_log_folder = os.path.join(
        args["save_pth"], "logs", "test", "epoch_{}".format(epoch)
    )
    os.makedirs(save_log_folder, exist_ok=True)
    os.makedirs(save_log_folder, exist_ok=True)

    bar = Bar("Processing", max=len(eval_loader))
    chamfer_distances = []
    chamfer_center_distances = []

    for step, (image_crop, obj_verts_gt, handverts_gt, handjoint_gt) in enumerate(eval_loader):
        obj_verts_gt = obj_verts_gt.float().cuda()
        handverts_gt = handverts_gt.float().cuda()
        handjoint_gt = handjoint_gt.float().cuda()
        # predict
        (
            pred_handverts, pred_handjoint, pred_shape, pred_pose,
            obj_verts, trans, scale
        ) = model(image_crop)
        # loss
        (
            obj_loss, hand_loss, contact_loss, total_loss
        ) = loss_model.compute_loss(
            obj_verts, trans, scale, obj_verts_gt,
            pred_handverts, pred_handjoint, pred_shape, pred_pose,
            handverts_gt, handjoint_gt,
            hand_faces, atlas_mesh
        )
        model_losses = {}
        model_losses["total_loss"] = total_loss
        model_losses["object_loss"] = obj_loss
        model_losses["hand_loss"] = hand_loss
        model_losses["contact_loss"] = torch.tensor(contact_loss)
        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)
        # eval
        _, _, cd, cdc = compute_chamfer(obj_verts_gt, obj_verts)
        if not math.isnan(cd):
            chamfer_distances.append(cd)
        if not math.isnan(cdc):
            chamfer_center_distances.append(cdc)
        # display
        save_img_path = os.path.join(
            save_img_folder, "img_step{}.png".format(step)
        )      
        save_log_path = os.path.join(
            save_log_folder, "log_step{}.json".format(step)
        )  
        if args["display_test_freq"] != 0 and step % args["display_test_freq"] == 0:
            image_crop = image_crop.permute(0, 2, 3, 1)
            # mask_crop = torch.cat([mask_gt.unsqueeze(-1), pred_mask.unsqueeze(-1).detach().cpu(), blue_channel.unsqueeze(-1)], dim=-1) #[128, 128, 3]
            displaymano.visualize_batch(
                args,
                "test",
                save_img_path,
                fig = fig,
                images = image_crop,
                # mask_crop=mask_crop,
                # obj
                objpoints3d_gt = obj_verts_gt,
                preds_objverts = obj_verts,
                obj_coarse_pc = obj_verts,
                # mask_pc=mask_points,
                # hand
                preds_handjoints = pred_handjoint,
                preds_handverts = pred_handverts,
                handjoints_3d_gt = handjoint_gt,
                # faces
                preds_objface = atlas_mesh,
                hand_faces=hand_faces,
                # camera=camera
            )  
            write_log(
                mesh_loss=avg_meters.average_meters["total_loss"].avg,
                scale_loss=avg_meters.average_meters["object_loss"].avg,
                hand_loss=avg_meters.average_meters["hand_loss"].avg,
                contact_loss=avg_meters.average_meters["contact_loss"].avg,
                CDC=sum(chamfer_distances) / len(chamfer_distances),
                CD=sum(chamfer_center_distances) / len(chamfer_center_distances),
                path=save_log_path
            )
        # bar
        bar.suffix = "({batch}/{size}) Total: {total_loss:.2f} | Param: {obj_loss:.2f} | Hand: {hand_loss:.2f} | Contact: {contact_loss:.2f} | CDC: {chamfer_distance_center:.2f} | CD: {chamfer_distance:.2f}".format(
            batch=step + 1,
            size=len(eval_loader),
            total_loss = avg_meters.average_meters["total_loss"].avg,
            obj_loss = avg_meters.average_meters["object_loss"].avg,
            hand_loss = avg_meters.average_meters["hand_loss"].avg,
            contact_loss = avg_meters.average_meters["contact_loss"].avg,
            chamfer_distance=sum(chamfer_distances) / len(chamfer_distances),
            chamfer_distance_center=sum(chamfer_center_distances) / len(chamfer_center_distances)
        )
        bar.next()  
    bar.finish()

    return sum(chamfer_center_distances) / len(chamfer_center_distances)