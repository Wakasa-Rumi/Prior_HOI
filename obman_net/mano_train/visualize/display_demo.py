from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from obman_net.handobjectdatasets.viz3d import visualize_joints_3d
from obman_net.handobjectdatasets.viz2d import visualize_joints_2d
import os

def create_segments(contact_infos, mesh_verts):
    missed_mask, penetr_mask, close_verts = contact_infos
    penetrating_verts = mesh_verts[penetr_mask == 1]
    penetrating_close_verts = close_verts[penetr_mask == 1]
    missed_verts = mesh_verts[missed_mask == 1]
    missed_close_verts = close_verts[missed_mask == 1]
    return penetrating_verts, penetrating_close_verts, missed_verts, missed_close_verts


def visualize_contacts3d(ax, contact_infos, mesh_verts, alpha=0.1):
    penetrating_verts, penetrating_close_verts, missed_verts, missed_close_verts = create_segments(
        contact_infos, mesh_verts
    )
    for penetrating_vert, close_vert in zip(penetrating_verts, penetrating_close_verts):
        ax.plot(
            [penetrating_vert[0], close_vert[0]],
            [penetrating_vert[1], close_vert[1]],
            [penetrating_vert[2], close_vert[2]],
            c="r",
            alpha=alpha,
        )
    for missed_vert, close_vert in zip(missed_verts, missed_close_verts):
        ax.plot(
            [missed_vert[0], close_vert[0]],
            [missed_vert[1], close_vert[1]],
            [missed_vert[2], close_vert[2]],
            c="b",
            alpha=alpha,
        )


def visualize_contacts2d(
    ax, contact_infos, mesh_verts, proj="z", contact_alpha=0.5, penetr_alpha=0.5
):
    proj_1, proj_2 = get_proj_axis(proj=proj)
    penetrating_verts, penetrating_close_verts, missed_verts, missed_close_verts = create_segments(
        contact_infos, mesh_verts
    )
    for penetrating_vert, close_vert in zip(penetrating_verts, penetrating_close_verts):
        ax.plot(
            [penetrating_vert[proj_1], close_vert[proj_1]],
            [penetrating_vert[proj_2], close_vert[proj_2]],
            c="r",
            alpha=penetr_alpha,
        )
    for missed_vert, close_vert in zip(missed_verts, missed_close_verts):
        ax.plot(
            [missed_vert[proj_1], close_vert[proj_1]],
            [missed_vert[proj_2], close_vert[proj_2]],
            c="b",
            alpha=contact_alpha,
        )


def visualize_batch(
    save_img_path,
    images,
    objpoints3d_gt,
    handjoints_3d_gt,
    preds_objverts,
    preds_objface,
    preds_handjoints,
    preds_handverts,
    faces_left,
    faces_right,
    fig=None,
    max_rows=4,
    side=None,
    obj_coarse_gt=None,
    obj_coarse_pc=None,
    gt_Mesh=None,
    idx=None,
):
    batch_nb = min(images.shape[0], max_rows)

    # Get hand joints
    if handjoints_3d_gt is not None:
        gt_batchjoints3d = handjoints_3d_gt.cpu().detach().numpy()
    pred_batchjoints3d = preds_handjoints.cpu().detach().numpy()
    pred_batchverts3d = preds_handverts.cpu().detach().numpy()

    # Get object vertices
    gt_batchobjpoints3d = objpoints3d_gt.detach().cpu().numpy()
    obj_coarse_pc = obj_coarse_pc.detach().cpu().numpy()
    obj_coarse_gt = obj_coarse_gt.detach().cpu().numpy()

    pred_batchobjpoints3d = preds_objverts.detach().cpu().numpy()
    pred_objfaces = preds_objface

    # contact
    has_contacts = False

    # Initialize figure
    row_factor = 1
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    fig.clf()
    col_nb = 5
    has_segm = False

    sides = None
    gt_batchjoints2d = None
    pred_batchjoints2d = None

    # Create figure
    for row_idx in range(batch_nb):
        # Show input image
        if sides is not None:
            side = sides[row_idx]
        else:
            side = None
        if images is not None:
            input_img = images[row_idx].numpy() + 0.5
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 1
            )
            ax.imshow(input_img)
            if gt_batchjoints2d is not None:
                gt_joints2d = gt_batchjoints2d[row_idx]
                visualize_joints_2d(ax, gt_joints2d, joint_idxs=False, alpha=0.5)
            if pred_batchjoints2d is not None:
                pred_joints2d = pred_batchjoints2d[row_idx]
                visualize_joints_2d(ax, pred_joints2d, joint_idxs=False)
            if side is not None:
                ax.set_title(side)
            ax.axis("off")

        # Get sample infos
        if handjoints_3d_gt is not None:
            gt_joints3d = get_row(gt_batchjoints3d, row_idx)
        pred_joints3d = get_row(pred_batchjoints3d, row_idx)
        pred_objpoints3d = get_row(pred_batchobjpoints3d, row_idx)
        verts3d = get_row(pred_batchverts3d, row_idx)
        gt_objpoints3d = get_row(gt_batchobjpoints3d, row_idx)

        rot_obj_coarse_pc = get_row(obj_coarse_pc, row_idx)
        gt_obj_coarse_pc = get_row(obj_coarse_gt, row_idx)
        idx_row = get_row(idx, row_idx)

        # Show output mesh
        ax = fig.add_subplot(
            batch_nb * row_factor,
            col_nb,
            row_idx * row_factor * col_nb + 2,
            projection="3d",
        )
        add_hand_obj_meshes(
            ax,
            verts3d=None,
            faces_right=None,
            gt_objpoints3d=None,
            pred_objpoints3d=pred_objpoints3d,
            pred_objfaces=None,
        )

        # Show x, y projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 3
        )
        add_joints_proj(ax, pred_joints3d, pred_joints3d, proj="z")
        # add_scatter_proj(ax, gt_objpoints3d, pred_objpoints3d, proj="z")
        ax.invert_yaxis()
        # ax = fig.add_subplot(
        #     batch_nb * row_factor, 
        #     col_nb, 
        #     row_idx * row_factor * col_nb + 3,
        #     projection="3d",
        # )
        # add_hand_obj_meshes1(
        #     ax,
        #     pred_objpoints3d=pred_objpoints3d,
        # )

        # Show x, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor,
            col_nb,
            row_idx * row_factor * col_nb + 4,
            projection="3d",
        )
        add_hand_obj_meshes(
            ax,
            verts3d=verts3d,
            faces_right=faces_right,
            gt_objpoints3d=None,
            pred_objpoints3d=None,
            pred_objfaces=None,
        )

        # Show y, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor,
            col_nb,
            row_idx * row_factor * col_nb + 5,
            projection="3d",
        )
        add_hand_obj_meshes(
            ax,
            verts3d=None,
            faces_right=None,
            gt_objpoints3d=None,
            pred_objpoints3d=gt_objpoints3d,
            pred_objfaces=None,
        )
    plt.savefig(save_img_path, dpi=100)

def visualize_demo(
    save_img_path,
    images,
    preds_objverts,
    preds_objface,
    preds_handjoints,
    preds_handverts,
    faces_right,
    fig=None,
    max_rows=4,
    side=None,
    obj_coarse_pc=None
):
    batch_nb = min(images.shape[0], max_rows)

    # Get hand joints
    pred_batchjoints3d = preds_handjoints.cpu().detach().numpy()
    pred_batchverts3d = preds_handverts.cpu().detach().numpy()

    # Get object vertices
    obj_coarse_pc = obj_coarse_pc.detach().cpu().numpy()

    pred_batchobjpoints3d = preds_objverts.detach().cpu().numpy()
    pred_objfaces = preds_objface

    # contact
    has_contacts = False

    # Initialize figure
    row_factor = 1
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    fig.clf()
    col_nb = 5
    has_segm = False

    sides = None
    gt_batchjoints2d = None
    pred_batchjoints2d = None

    # Create figure
    for row_idx in range(batch_nb):
        # Show input image
        if sides is not None:
            side = sides[row_idx]
        else:
            side = None
        if images is not None:
            input_img = images[row_idx].numpy() + 0.5
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 1
            )
            ax.imshow(input_img)
            if gt_batchjoints2d is not None:
                gt_joints2d = gt_batchjoints2d[row_idx]
                visualize_joints_2d(ax, gt_joints2d, joint_idxs=False, alpha=0.5)
            if pred_batchjoints2d is not None:
                pred_joints2d = pred_batchjoints2d[row_idx]
                visualize_joints_2d(ax, pred_joints2d, joint_idxs=False)
            if side is not None:
                ax.set_title(side)
            ax.axis("off")

        # Get sample infos
        pred_joints3d = get_row(pred_batchjoints3d, row_idx)
        pred_objpoints3d = get_row(pred_batchobjpoints3d, row_idx)
        verts3d = get_row(pred_batchverts3d, row_idx)

        # Show output mesh
        ax = fig.add_subplot(
            batch_nb * row_factor,
            col_nb,
            row_idx * row_factor * col_nb + 2,
            projection="3d",
        )
        add_hand_obj_meshes(
            ax,
            verts3d=None,
            faces_right=None,
            pred_objpoints3d=pred_objpoints3d,
            pred_objfaces=None,
        )

        # Show x, y projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 3
        )
        add_joints_proj(ax, None, pred_joints3d, proj="z")
        add_scatter_proj(ax, None, pred_objpoints3d, proj="z")
        ax.invert_yaxis()

        # Show x, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 4
        )
        add_joints_proj(ax, None, pred_joints3d, proj="y")
        add_scatter_proj(ax, None, pred_objpoints3d, proj="y")
        ax.invert_yaxis()

        # Show y, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 5
        )
        add_joints_proj(ax, None, pred_joints3d, proj="x")
        add_scatter_proj(ax, None, pred_objpoints3d, proj="x")
        ax.invert_yaxis()
    plt.savefig(save_img_path, dpi=100)

def get_row(batch_sample, idx):
    if batch_sample is not None:
        row_sample = batch_sample[idx]
    else:
        row_sample = None
    return row_sample

def add_hand_obj_meshes(
    ax,
    verts3d=None,
    faces_right=None,
    gt_objpoints3d=None,
    pred_objpoints3d=None,
    pred_objfaces=None,
):

    # visualize_joints_3d(ax, gt_batchjoints3d, joint_idxs=joint_idxs)
    # Add mano predictions
    if faces_right is not None:
        add_mesh(ax, verts3d, faces_right) 

    # Add object gt/predictions
    if pred_objpoints3d is not None:
        if pred_objfaces is not None:
            add_mesh(ax, pred_objpoints3d, pred_objfaces, c="r")
        else:
            ax.scatter(
                pred_objpoints3d[:, 0],
                pred_objpoints3d[:, 1],
                pred_objpoints3d[:, 2],
                c="r",
                alpha=0.2,
                s=3,
            )
    if gt_objpoints3d is not None:
        ax.scatter(
            gt_objpoints3d[:, 0],
            gt_objpoints3d[:, 1],
            gt_objpoints3d[:, 2],
            c="g",
            alpha=0.2,
            s=1,
        )
    else:
        gt_objpoints3d = None
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def add_hand_obj_meshes1(
    ax,
    verts3d=None,
    faces_right=None,
    gt_objpoints3d=None,
    pred_objpoints3d=None,
    pred_objfaces=None,
):

    # visualize_joints_3d(ax, gt_batchjoints3d, joint_idxs=joint_idxs)
    # Add mano predictions
    if faces_right is not None:
        add_mesh(ax, verts3d, faces_right) 

    # Add object gt/predictions
    if pred_objpoints3d is not None:
        if pred_objfaces is not None:
            add_mesh(ax, pred_objpoints3d, pred_objfaces, c="r")
        else:
            ax.scatter(
                pred_objpoints3d[:, 1],
                pred_objpoints3d[:, 2],
                pred_objpoints3d[:, 0],
                c="r",
                alpha=0.2,
                s=3,
            )
    if gt_objpoints3d is not None:
        ax.scatter(
            gt_objpoints3d[:, 1],
            gt_objpoints3d[:, 2],
            gt_objpoints3d[:, 0],
            c="g",
            alpha=0.2,
            s=1,
        )
    else:
        gt_objpoints3d = None
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def get_proj_axis(proj="z"):
    if proj == "z":
        proj_1 = 0
        proj_2 = 1
    elif proj == "y":
        proj_1 = 0
        proj_2 = 2
    elif proj == "x":
        proj_1 = 1
        proj_2 = 2
    return proj_1, proj_2


def add_scatter_proj(ax, gt_objpoints3d, pred_objpoints3d, proj="z"):
    proj_1, proj_2 = get_proj_axis(proj=proj)
    if pred_objpoints3d is not None:
        ax.scatter(
            pred_objpoints3d[:, proj_1],
            pred_objpoints3d[:, proj_2],
            c="r",
            alpha=0.1,
            s=1,
        )
    if gt_objpoints3d is not None:
        ax.scatter(
            gt_objpoints3d[:, proj_1], gt_objpoints3d[:, proj_2], c="g", alpha=0.1, s=1
        )
    ax.set_aspect("equal")


def add_joints_proj(ax, gt_keypoints, pred_keypoints, proj="z", joint_idxs=False):
    proj_1, proj_2 = get_proj_axis(proj=proj)
    if gt_keypoints is not None:
        visualize_joints_2d(
            ax,
            np.stack([gt_keypoints[:, proj_1], gt_keypoints[:, proj_2]], axis=1),
            alpha=0.2,
            joint_idxs=joint_idxs,
        )
    if pred_keypoints is not None:
        visualize_joints_2d(
            ax,
            np.stack([pred_keypoints[:, proj_1], pred_keypoints[:, proj_2]], axis=1),
            joint_idxs=joint_idxs,
        )
    ax.set_aspect("equal")


def add_mesh(ax, verts, faces, flip_x=False, c="b", alpha=0.1):
    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    plt.tight_layout()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

    # ax.set_xlim(-100, 100)
    # # Invert y and z axis
    # ax.set_ylim(100, -100)
    # ax.set_zlim(centers[2] + r, centers[2] - r)


def save_pck_img(thresholds, pck_values, auc_all, save_pck_file, overlay=None):
    """
    Args:
        auc_all (float): Area under the curve
    """
    plt.clf()
    fontsize = 36
    markersize = 16
    plt.plot(thresholds, pck_values, "ro-", markersize=markersize, label="Ours")

    # Relative 3D for 12 sequences of stereohands
    if overlay == "stereo_all":
        plt.title("Stereo dataset (12 seq.)", fontsize=40)
        gan_thresh = [
            20.2020,
            22.2222,
            24.2424,
            26.2626,
            28.2828,
            30.3030,
            32.3232,
            34.3434,
            36.3636,
            38.3838,
            40.4040,
            42.4242,
            44.4444,
            46.4646,
            48.4848,
            50.5051,
        ]

        gan_accuracies = [
            0.4416,
            0.4772,
            0.5101,
            0.5410,
            0.5699,
            0.5968,
            0.6212,
            0.6445,
            0.6660,
            0.6858,
            0.7049,
            0.7229,
            0.7394,
            0.7550,
            0.7697,
            0.7835,
        ]
        plt.plot(
            gan_thresh, gan_accuracies, "bv-", markersize=markersize, label="Ganerated"
        )

    # Relative 3D for 2 sequences of stereohands
    elif overlay == "stereo_test":
        plt.title("Stereo dataset (2 seq.)", fontsize=40)
        gan_thresh = [
            19.1919,
            22.2222,
            25.2525,
            28.2828,
            31.3131,
            34.3434,
            37.3737,
            40.4040,
            43.4343,
            46.4646,
            49.4949,
        ]
        gan_accuracies_wo = [
            0.7031,
            0.7323,
            0.7586,
            0.7831,
            0.8056,
            0.8249,
            0.8424,
            0.8586,
            0.8728,
            0.8859,
            0.8972,
        ]
        gan_accuracies_w = [
            0.8713,
            0.9035,
            0.9271,
            0.9446,
            0.9574,
            0.9670,
            0.9741,
            0.9795,
            0.9833,
            0.9867,
            0.9895,
        ]
        plt.plot(
            gan_thresh,
            gan_accuracies_wo,
            "bv-",
            markersize=markersize,
            label="Ganerated wo",
        )
        plt.plot(
            gan_thresh,
            gan_accuracies_w,
            "c^-",
            markersize=markersize,
            label="Ganerated w",
        )
        zimmerman_thresh = [
            21.0526315789474,
            23.6842105263158,
            26.3157894736842,
            28.9473684210526,
            31.5789473684211,
            34.2105263157895,
            36.8421052631579,
            39.4736842105263,
            42.1052631578947,
            44.7368421052632,
            47.3684210526316,
            50,
        ]
        zimmerman_accs = [
            0.869888888888889,
            0.896873015873016,
            0.916849206349206,
            0.932142857142857,
            0.943507936507937,
            0.952753968253968,
            0.959904761904762,
            0.966047619047619,
            0.971595238095238,
            0.976547619047619,
            0.980174603174603,
            0.983277777777778,
        ]
        plt.plot(
            zimmerman_thresh, zimmerman_accs, "gs-", markersize=markersize, label="Z&B"
        )
        chpr_thresh = [20, 25, 30, 35, 40, 45, 50]
        chpr_accs = [
            0.565789473684211,
            0.717105263157895,
            0.822368421052632,
            0.881578947368421,
            0.914473684210526,
            0.9375,
            0.960526315789474,
        ]
        plt.plot(chpr_thresh, chpr_accs, "mD-", markersize=markersize, label="CHPR")

    else:
        plt.title(
            "auc in [{},{}]: {}".format(thresholds[0], thresholds[-1], auc_all),
            fontsize=40,
        )
    plt.ylim(0, 1)
    plt.xlabel("Error Thresholds (mm)", fontsize=fontsize)
    plt.ylabel("3D PCK", fontsize=fontsize)
    plt.grid(linestyle="-", color="lightgray", alpha=0.5)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_pck_file, format="eps")
