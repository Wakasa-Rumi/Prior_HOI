import pickle
import os
from progress.bar import Bar as Bar

from obman_net.mano_train.evaluation.evalutils import AverageMeters
from obman_net.mano_train.evaluation.zimeval import EvalUtil
from obman_net.mano_train.visualize import displaymano
from hand_pipline.YTBHand_network import Pool
from loss.acc import test_hand_acc, test_obj_acc, compute_chamfer

def epoch_dexycb_train(
    train_loader,
    model,
    loss_model,
    epoch,
    optimizer,
    hand_tmp,
    args,
    fig,
    down_transform_list=None
):
    avg_meters = AverageMeters()
    with open("obman_net/misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        hand_faces = mano_right_data["f"]
    atlas_mesh = model.module.test_faces
    
    model.eval()
    save_img_folder = os.path.join(
        args["save_pth"], "images", "train", "epoch_{}".format(epoch)
    )
    os.makedirs(save_img_folder, exist_ok=True)

    bar = Bar("Processing", max=len(train_loader))
    chamfer_distances = []
    chamfer_center_distances = []

    for step, (image_crop, obj_point_gt, obj_coarse_pc, handverts_gt, handjoint_gt) in enumerate(train_loader):
        handverts_gt = handverts_gt.float().cuda()
        handjoint_gt = handjoint_gt.float().cuda()
        obj_point_gt = obj_point_gt.float().cuda()
        obj_coarse_pc = obj_coarse_pc.float().cuda()
        
        if args["model_name"] == "HOInet" or args["model_name"] == "DGhoi" or args["model_name"] == "PoseHOI":
            batch_vertices_all = []
            batch_vertices_all.append(handverts_gt)#B 778 3;B 389 3;B 195 3;B 98 3
            for i in range(len(down_transform_list)-1):
                verts = Pool(handverts_gt, down_transform_list[i].cuda())
                batch_vertices_all.append(verts)
            (
                pred_verts, pred_trans, pred_scale, 
                hand_verts_list, hand_joint,
                refine_obj
            ) = model.forward(image_crop, obj_coarse_pc)
            hand_verts = hand_verts_list[-1]
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                args, hand_tmp,
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts_list, hand_joint, batch_vertices_all, handjoint_gt,
                hand_faces, atlas_mesh,
                refine_obj
            )
        elif args["model_name"] == "Obman":
            (
                hand_verts, hand_joint, hand_shape, hand_pose,
                pred_verts, pred_trans, pred_scale
            ) = model.forward(image_crop)
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts, hand_joint, hand_shape, hand_pose,
                handverts_gt, handjoint_gt,
                hand_faces, atlas_mesh
            )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()      

        # Get values out of tensors
        model_losses = {}
        model_losses["total_loss"] = total_loss
        model_losses["atlas_loss"] = atlas_loss
        model_losses["hand_loss"] = hand_loss
        model_losses["refine_loss"] = refine_loss
        model_losses["contact_loss"] = contact_loss
        for key, val in model_losses.items():
            if val is not None:
               avg_meters.add_loss_value(key, val)

        # Get Chamfer distance
        if args["refine"] == False:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, pred_verts)
        else:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, refine_obj)
        chamfer_distances.append(cd)
        chamfer_center_distances.append(cdc)

        # display
        save_img_path = os.path.join(
            save_img_folder, "img_step{}.png".format(step)
        )       
        image_crop = image_crop.permute(0, 2, 3, 1)
        if args["display_train_freq"] != 0 and step % args["display_train_freq"] == 0:
            if args["refine"] == False:
                show_prior = obj_coarse_pc.permute(0, 2, 1)
                show_pred = pred_verts
            else:
                show_prior = pred_verts
                show_pred = refine_obj                
            displaymano.visualize_batch(
                args,
                "train",
                save_img_path,
                fig = fig,
                images = image_crop,
                # obj
                objpoints3d_gt = obj_point_gt,
                preds_objverts = show_pred,
                obj_coarse_pc = show_prior,
                # hand
                preds_handjoints = hand_joint,
                preds_handverts = hand_verts_list[-1],
                handjoints_3d_gt = handjoint_gt,
                # faces
                preds_objface = atlas_mesh,
                hand_faces=hand_faces,
            )        

        # bar
        bar.suffix = "({batch}/{size}) Atlas: {atlas_total_loss:.2f} | Hand: {hand_loss:.2f} | Contact: {contact_loss:.2f} | CDC: {chamfer_distance_center:.4f} | CD: {chamfer_distance:.4f} ".format(
            batch=step + 1,
            size=len(train_loader),
            atlas_total_loss=avg_meters.average_meters["atlas_loss"].avg,
            hand_loss=avg_meters.average_meters["hand_loss"].avg,
            contact_loss=avg_meters.average_meters["contact_loss"].avg,
            chamfer_distance=sum(chamfer_distances) / len(chamfer_distances),
            chamfer_distance_center=sum(chamfer_center_distances) / len(chamfer_center_distances)
        )
        bar.next()          
    
    bar.finish()

def epoch_dexycb_val(
    eval_loader,
    model,
    loss_model,
    epoch,
    hand_tmp,
    args,
    fig,
    down_transform_list=None
):
    avg_meters = AverageMeters()
    with open("obman_net/misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        hand_faces = mano_right_data["f"]
    atlas_mesh = model.module.test_faces
    
    model.eval()
    save_img_folder = os.path.join(
        args["save_pth"], "images", "val", "epoch_{}".format(epoch)
    )
    os.makedirs(save_img_folder, exist_ok=True)

    bar = Bar("Processing", max=len(eval_loader))
    chamfer_distances = []
    chamfer_center_distances = []

    for step, (image_crop, obj_point_gt, obj_coarse_pc, handverts_gt, handjoint_gt) in enumerate(eval_loader):
        handverts_gt = handverts_gt.float().cuda()
        handjoint_gt = handjoint_gt.float().cuda()
        obj_point_gt = obj_point_gt.float().cuda()
        obj_coarse_pc = obj_coarse_pc.float().cuda()
        
        if args["model_name"] == "HOInet" or args["model_name"] == "DGhoi" or args["model_name"] == "PoseHOI":
            batch_vertices_all = []
            batch_vertices_all.append(handverts_gt)#B 778 3;B 389 3;B 195 3;B 98 3
            for i in range(len(down_transform_list)-1):
                verts = Pool(handverts_gt, down_transform_list[i].cuda())
                batch_vertices_all.append(verts)
            (
                pred_verts, pred_trans, pred_scale, 
                hand_verts_list, hand_joint,
                refine_obj
            ) = model.forward(image_crop, obj_coarse_pc)
            hand_verts = hand_verts_list[-1]
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                args, hand_tmp,
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts_list, hand_joint, batch_vertices_all, handjoint_gt,
                hand_faces, atlas_mesh,
                refine_obj
            )
        elif args["model_name"] == "Obman":
            (
                hand_verts, hand_joint, hand_shape, hand_pose,
                pred_verts, pred_trans, pred_scale
            ) = model.forward(image_crop)
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts, hand_joint, hand_shape, hand_pose,
                handverts_gt, handjoint_gt,
                hand_faces, atlas_mesh
            )

        # Get values out of tensors
        model_losses = {}
        model_losses["total_loss"] = total_loss
        model_losses["atlas_loss"] = atlas_loss
        model_losses["hand_loss"] = hand_loss
        model_losses["refine_loss"] = refine_loss
        model_losses["contact_loss"] = contact_loss
        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        # Get Chamfer distance
        if args["refine"] == False:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, pred_verts)
        else:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, refine_obj)
        chamfer_distances.append(cd)
        chamfer_center_distances.append(cdc)

        # display
        save_img_path = os.path.join(
            save_img_folder, "img_step{}.png".format(step)
        )       
        image_crop = image_crop.permute(0, 2, 3, 1)
        if args["display_val_freq"] != 0 and step % args["display_val_freq"] == 0:
            if args["refine"] == False:
                show_prior = obj_coarse_pc.permute(0, 2, 1)
                show_pred = pred_verts
            else:
                show_prior = pred_verts
                show_pred = refine_obj                
            displaymano.visualize_batch(
                args,
                "val",
                save_img_path,
                fig = fig,
                images = image_crop,
                # obj
                objpoints3d_gt = obj_point_gt,
                preds_objverts = show_pred,
                obj_coarse_pc = show_prior,
                # hand
                preds_handjoints = hand_joint,
                preds_handverts = hand_verts_list[-1],
                handjoints_3d_gt = handjoint_gt,
                # faces
                preds_objface = atlas_mesh,
                hand_faces=hand_faces,
            )        

        # bar
        bar.suffix = "({batch}/{size}) Atlas: {atlas_total_loss:.2f} | Hand: {hand_loss:.2f} | Contact: {contact_loss:.2f} | CDC: {chamfer_distance_center:.4f} | CD: {chamfer_distance:.4f} ".format(
            batch=step + 1,
            size=len(eval_loader),
            atlas_total_loss=avg_meters.average_meters["atlas_loss"].avg,
            hand_loss=avg_meters.average_meters["hand_loss"].avg,
            contact_loss=avg_meters.average_meters["contact_loss"].avg,
            chamfer_distance=sum(chamfer_distances) / len(chamfer_distances),
            chamfer_distance_center=sum(chamfer_center_distances) / len(chamfer_center_distances)
        )
        bar.next()          
    
    bar.finish()
    return sum(chamfer_distances) / len(chamfer_distances)

def epoch_dexycb_test(
    test_loader,
    model,
    loss_model,
    epoch,
    hand_tmp,
    args,
    fig,
    down_transform_list=None
):
    avg_meters = AverageMeters()
    with open("obman_net/misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        hand_faces = mano_right_data["f"]
    atlas_mesh = model.module.test_faces
    
    model.eval()
    save_img_folder = os.path.join(
        args["save_pth"], "images", "test", "epoch_{}".format(epoch)
    )
    os.makedirs(save_img_folder, exist_ok=True)

    bar = Bar("Processing", max=len(test_loader))
    chamfer_distances = []
    chamfer_center_distances = []

    evaluator_hand = EvalUtil()
    evaluator_obj = EvalUtil()

    for step, (image_crop, obj_point_gt, obj_coarse_pc, handverts_gt, handjoint_gt) in enumerate(test_loader):
        handverts_gt = handverts_gt.float().cuda()
        handjoint_gt = handjoint_gt.float().cuda()
        obj_point_gt = obj_point_gt.float().cuda()
        obj_coarse_pc = obj_coarse_pc.float().cuda()
        
        if args["model_name"] == "HOInet" or args["model_name"] == "DGhoi" or args["model_name"] == "PoseHOI":
            batch_vertices_all = []
            batch_vertices_all.append(handverts_gt)#B 778 3;B 389 3;B 195 3;B 98 3
            for i in range(len(down_transform_list)-1):
                verts = Pool(handverts_gt, down_transform_list[i].cuda())
                batch_vertices_all.append(verts)
            (
                pred_verts, pred_trans, pred_scale, 
                hand_verts_list, hand_joint,
                refine_obj
            ) = model.forward(image_crop, obj_coarse_pc)
            hand_verts = hand_verts_list[-1]
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                args, hand_tmp,
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts_list, hand_joint, batch_vertices_all, handjoint_gt,
                hand_faces, atlas_mesh,
                refine_obj
            )
        elif args["model_name"] == "Obman":
            (
                hand_verts, hand_joint, hand_shape, hand_pose,
                pred_verts, pred_trans, pred_scale
            ) = model.forward(image_crop)
            # get loss
            (
                atlas_loss, hand_loss, contact_loss, 
                refine_loss, total_loss 
            ) = loss_model.compute_loss(
                pred_verts, pred_trans, pred_scale, obj_point_gt,
                hand_verts, hand_joint, hand_shape, hand_pose,
                handverts_gt, handjoint_gt,
                hand_faces, atlas_mesh
            )

        # Get values out of tensors
        model_losses = {}
        model_losses["hand_loss"] = hand_loss
        model_losses["contact_loss"] = contact_loss
        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        # Get Chamfer distance
        if args["refine"] == False:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, pred_verts)
        else:
            _, _, cd, cdc = compute_chamfer(obj_point_gt, refine_obj)
        chamfer_distances.append(cd)
        chamfer_center_distances.append(cdc)

        # display
        save_img_path = os.path.join(
            save_img_folder, "img_step{}.png".format(step)
        )       
        image_crop = image_crop.permute(0, 2, 3, 1)
        if args["display_test_freq"] != 0 and step % args["display_test_freq"] == 0:
            if args["refine"] == False:
                show_prior = obj_coarse_pc.permute(0, 2, 1)
                show_pred = pred_verts
            else:
                show_prior = pred_verts
                show_pred = refine_obj                
            displaymano.visualize_batch(
                args,
                "test",
                save_img_path,
                fig = fig,
                images = image_crop,
                # obj
                objpoints3d_gt = obj_point_gt,
                preds_objverts = show_pred,
                obj_coarse_pc = show_prior,
                # hand
                preds_handjoints = hand_joint,
                preds_handverts = hand_verts_list[-1],
                handjoints_3d_gt = handjoint_gt,
                # faces
                preds_objface = atlas_mesh,
                hand_faces=hand_faces,
            )        

        # compute pck
        for gt_kp, pred_kp in zip(handjoint_gt.detach().cpu(), hand_joint.detach().cpu()):
            evaluator_hand.feed(gt_kp, pred_kp)
        for gt_kp, pred_kp in zip(obj_point_gt, pred_verts):
            evaluator_obj.feed_object(gt_kp, pred_kp)

        # bar
        bar.suffix = "({batch}/{size})  Hand: {hand_loss:.2f} | Contact:{contact_loss:.2f} | CDC: {chamfer_distance_center:.4f} | CD: {chamfer_distance:.4f} ".format(
            batch=step + 1,
            size=len(test_loader),
            hand_loss=avg_meters.average_meters["hand_loss"].avg,
            contact_loss=avg_meters.average_meters["contact_loss"].avg,
            chamfer_distance=sum(chamfer_distances) / len(chamfer_distances),
            chamfer_distance_center=sum(chamfer_center_distances) / len(chamfer_center_distances)
        )
        bar.next()          

    pck_info_hand = test_hand_acc(evaluator_hand)
    pck_info_10, pck_info_5 = test_obj_acc(evaluator_obj)
    
    bar.finish()
    return pck_info_hand, pck_info_10, pck_info_5