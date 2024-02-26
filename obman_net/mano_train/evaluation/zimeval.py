import numpy as np
import torch
from obman_net.mano_train.networks.branches import atlasutils
import trimesh

#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

def compute_chamfer(obj_point_input, pred_verts):
    c_loss = atlasutils.ChamferLoss()
    # print(obj_point_input.device)
    # print(pred_verts.device)

    # gt_verts_center = obj_point_input.mean(dim=1).unsqueeze(1)
    # pred_verts_center = pred_verts.mean(dim=1).unsqueeze(1)
    # c_gt = obj_point_input - gt_verts_center
    # c_pr = pred_verts - pred_verts_center

    c_gt = obj_point_input
    c_pr = pred_verts

    d1, d2 = c_loss(c_gt, c_pr)
    chamfer_distance = d1 + d2

    chamfer_distance = torch.mean(chamfer_distance, dim=-1).item()
    return c_gt, c_pr, chamfer_distance

def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def intersect(obj_mesh, hand_mesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh

class EvalUtil:
    """ Util class for evaluation networks.
    """

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())
        
        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.data4 = []
        self.obj_num = [0, 0, 0, 0]

    def feed(self, keypoint_gt, keypoint_pred, keypoint_vis=None):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """
        if isinstance(keypoint_gt, torch.Tensor):
            keypoint_gt = keypoint_gt.numpy()
        if isinstance(keypoint_pred, torch.Tensor):
            keypoint_pred = keypoint_pred.numpy()
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)

        if keypoint_vis is None:
            keypoint_vis = np.ones_like(keypoint_gt[:, 0])
        keypoint_vis = np.squeeze(keypoint_vis).astype("bool")

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = (keypoint_gt - keypoint_pred) / 1000
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            self.data[i].append(euclidean_dist[i])

    def feed_object(self, obj_gt, obj_pred, keypoint_vis=None):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """
        obj_gt = obj_gt.unsqueeze(0)
        obj_pred = obj_pred.unsqueeze(0)
        c_gt, c_pr, chamfer_dist = compute_chamfer(obj_point_input=obj_gt, pred_verts=obj_pred)
        chamfer_dist = (chamfer_dist / 1000)
        # chamfer_dist = np.array(chamfer_dist)

        self.data[0].append(chamfer_dist)
    
    def feed_intersect(self, hand_verts, obj_verts, hand_faces, obj_faces):
        hand_mesh = trimesh.Trimesh(
            vertices=hand_verts, faces=hand_faces
        )
        obj_mesh = trimesh.Trimesh(
            vertices=obj_verts, faces=obj_faces
        )  
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005) 

        num_kp = hand_verts.shape[0]
        for i in range(num_kp):
            self.data[i].append(volume[i])           

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def _get_num_dis(self):
        data1 = np.array(self.data1)
        data2 = np.array(self.data2)
        data3 = np.array(self.data3)
        data4 = np.array(self.data4)

        return np.mean(data1), np.mean(data2), np.mean(data3), np.mean(data4)

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # n1, n2, n3, n4 = self._get_num_dis()
        # n_distance = [n1, n2, n3, n4]

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)
        # Display error per keypoint
        epe_mean_joint = epe_mean_all
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds,
            # n_distance,
            # self.obj_num
        )