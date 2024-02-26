from pickletools import uint8
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data as data
from PIL import Image
import os
from random import random

import cv2
import open3d as o3d
from pytorch3d.io import load_objs_as_meshes
from torchvision.transforms import functional as func_transforms
from obman_net.manopth_master.manopth.manolayer import ManoLayer
from dataloader.data_augmentation import axis_angle_t_to_matrix, initial_temp_rot

_YCB_CLASSES = {
    1: "010_potted_meat_can",
    2: "021_bleach_cleanser",
    3: "019_pitcher_base",
    4: "003_cracker_box",
    5: "006_mustard_bottle",
    6: "004_sugar_box",
    7: "035_power_drill",
    8: "011_banana",
    9: "037_scissors",
    10: "025_mug"
}

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (-xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def cut_img(img_list, mask_list, label2d_list, camera=None, radio=0.7, img_size=256):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]],
                                     [0, 1, L - mid[1]]])

    img_list_out = []
    for img in img_list:
        img_list_out.append(cv2.warpAffine(img, M, dsize=(img_size, img_size)))
    
    mask_list_out = []
    for mask in mask_list:
        mask_list_out.append(cv2.warpAffine(mask, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, mask_list_out, label2d_list_out, camera

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

def display_obj(verts, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True, num=0):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)

    cam_equal_aspect_3d(ax, verts)
    plt.savefig('obj.png')

class HO3DDataset(data.Dataset):
    def __init__(self, setup, split, model):
        ycb_classes = _YCB_CLASSES

        self._setup = setup
        self._split = split
        self.model = model

        self._data_dir = "/home/yiyao/HOI/datasets/HO3D_v3/"
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._train_dir = os.path.join(self._data_dir, "train")
        self._eval_dir = os.path.join(self._data_dir, "evaluation")
        self._rgb_format = "{:04d}.jpg"
        self._depth_format = "{:04d}.jpg"
        self._h = 480
        self._w = 640

        self._object_dir = os.path.join(self._data_dir, "cache/ho3d_vid_train_mesh.pkl")

        # load object template
        objects_point_all = {}
        f = open(self._object_dir, 'rb')
        obj_data = pickle.load(f)
        for i in range(1,11):
            verts_list = obj_data[ycb_classes[i]].verts_list
            verts = verts_list()[0]
            objects_point_all[ycb_classes[i]] = verts
        self.objects_point_all = objects_point_all
        self.obj_data = obj_data
    
        # load split file
        self.train_dir_list = []
        self.eval_dir_list = []

        file_train = open("/home/yiyao/HOI/datasets/HO3D_v3/train.txt", "rb")
        line_train = file_train.readline()
        while line_train:
            self.train_dir_list.append(line_train.strip().decode())
            line_train = file_train.readline()

        file_evaluation = open("/home/yiyao/HOI/datasets/HO3D_v3/evaluation.txt", "rb")
        line_evaluation = file_evaluation.readline()
        while line_evaluation:
            self.eval_dir_list.append(line_evaluation.strip().decode())
            line_evaluation = file_evaluation.readline()    

        # load projections
        self.project_dict = torch.load("preprocess/mask/dict.pkl")
        self.project_dict_gt = torch.load("preprocess/mask_gt/dict.pkl")
    
    def __len__(self):
        if self._split == 'training':
            return len(self.train_dir_list)
            # return 32
        if self._split == 'val' or self._split == 'test':
            return len(self.eval_dir_list)
            # return 32
    
    def __getitem__(self, idx):
        idx = idx
        if self._split == 'training':
            dir_num = self.train_dir_list[idx].split("/")
            root_dir = self._train_dir
        if self._split == 'val' or self._split == 'test':
            dir_num = self.eval_dir_list[idx].split("/")
            root_dir = self._eval_dir

        dir = dir_num[0]
        num = dir_num[1]
        rgb_dir = root_dir + "/" + dir + "/rgb/" + num + ".jpg"
        meta_dir = root_dir + "/" + dir + "/meta/" + num + ".pkl"
        mask_dir = root_dir + "/" + dir + "/mask/" + num + ".jpg"
        gt_mask_dir = root_dir + "/" + dir + "/gt_mask/" + num + ".jpg"

        # Load picture
        image = Image.open(rgb_dir).convert("RGB")
        mask = Image.open(mask_dir).convert("RGB")
        if self.model != "GenMask":
            mask_gt = np.array(Image.open(gt_mask_dir).convert("RGB"))
            mask_gt = torch.tensor(mask_gt / 255)[:,:,0]

        # Load meta data
        # handBeta, handTrans, handPose, handJoints3D, camMat, objRot,
        # objTrans, camIDList, objCorners3D, objCorners3DRest, objName,
        # objLabel, handVertContact, handVertDist, handVertIntersec, handVertObjSurfProj
        meta_file = open(meta_dir, "rb")
        meta_data = pickle.load(meta_file)

        # Load object
        object_name = meta_data['objName']
        d_pe = "preprocess/template/" + object_name + ".pth"
        d_mesh = "preprocess/template/" + object_name + ".obj"
        d_temp_mesh = "preprocess/meshes/template_" + object_name + ".obj"

        rot_meta = open("preprocess/rotated_prior/rot_matrix.pth", "rb")
        rot_p = pickle.load(rot_meta)[object_name]
        rot_meta_fine = open("preprocess/rotated_prior/fine_rot_matrix.pth", "rb")
        rot_fp = pickle.load(rot_meta_fine)[object_name]

        # Load prior Mesh
        prior_meshes = load_objs_as_meshes([d_mesh])
        prior_verts = prior_meshes.verts_packed() * 1000
        prior_faces = prior_meshes.faces_packed()

        temp_meshes = load_objs_as_meshes([d_temp_mesh])
        temp_verts = temp_meshes.verts_packed()
        temp_faces = temp_meshes.faces_packed()

        gt_prior_verts =  torch.matmul(torch.matmul(prior_verts, rot_p.transpose(0,1)), rot_fp.transpose(0,1))

        # Load aligned Mesh
        uni_meshes = o3d.io.read_triangle_mesh(d_mesh)
        uni_verts = o3d.geometry.TriangleMesh.sample_points_uniformly(uni_meshes, number_of_points=1024)
        uni_verts = torch.tensor(np.asarray(uni_verts.points)) * 1000
    
        # Load projection
        pro_xz_prior = self.project_dict[object_name]["xz"].unsqueeze(0)
        pro_xz_gt = self.project_dict_gt[object_name]["xz"].unsqueeze(0)
        pro_yz_prior = self.project_dict[object_name]["yz"].unsqueeze(0)
        pro_yz_gt = self.project_dict_gt[object_name]["yz"].unsqueeze(0)
        zeros = torch.zeros_like(pro_xz_prior)

        obj_point_model = self.objects_point_all[object_name]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_point_model)
        downpcd = pcd.farthest_point_down_sample(num_samples=1024)
        obj_point_model = np.asarray(downpcd.points)

        try:
            object_rot = axis_angle_t_to_matrix(torch.tensor(meta_data['objRot']).reshape(3))[:3,:3]
        except:
            print(meta_dir)

        if self.model == "GenMask":
            if not os.path.isdir(root_dir + "/" + dir + "/gt_mask/"):
                os.makedirs(root_dir + "/" + dir + "/gt_mask/")
            rotated_v = torch.tensor(np.matmul(temp_verts, object_rot.T))
            return rotated_v, temp_faces, gt_mask_dir

        obj_point = ((np.matmul(obj_point_model, object_rot.T)) + meta_data['objTrans']) * 1000

        gt_prior_verts = torch.matmul(gt_prior_verts, torch.tensor(object_rot.T))

        new_rot = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        shape_prior = torch.tensor(torch.load(d_pe)[:1024]) * 1000
        template = shape_prior

        # Load camera
        K = meta_data['camMat']

        # Load MANO layer
        if self._split == 'training':
            mano_layer = ManoLayer(
                flat_hand_mean=False,
                ncomps=45,
                side='right',
                mano_root='misc/mano',
                use_pca=True
            )
            hand_pose = torch.tensor(meta_data['handPose']).unsqueeze(0).float()
            hand_trans = torch.tensor(meta_data['handTrans']).unsqueeze(0).float()
            faces = mano_layer.th_faces.numpy()
            betas = torch.tensor(meta_data['handBeta']).unsqueeze(0).float()
            # Add MANO meshes
            vert, jtr = mano_layer(hand_pose, betas, hand_trans)
            jtr = jtr[0]
            vert = vert[0]

            # prior_verts[:, 0] = -prior_verts[:,0]
            # template[:,0] = -template[:,0]
            # obj_point[:,0] = -obj_point[:,0]
            # vert[:,0] = -vert[:,0]
            # jtr[:,0] = -jtr[:,0]

            # crop the image and generate the new K
            obj_point_uvd = xyz2uvd(obj_point, K)
            obj_point_2d = obj_point_uvd[:, :2]

            joint_uvd = xyz2uvd(jtr, K)
            joint_2d = joint_uvd[:, :2]

            image = np.array(image)
            img_list = [image]
            mask = np.array(mask)
            mask_list = [mask]
            label2d_list = [obj_point_2d]
        else:
            # template[:,0] = -template[:,0]
            # obj_point[:,0] = -obj_point[:,0]
            obj_point_uvd = xyz2uvd(obj_point, K)
            obj_point_2d = obj_point_uvd[:, :2]

            label2d_list = [obj_point_2d]
            image = np.array(image)
            img_list = [image]
            mask = np.array(mask)
            mask_list = [mask]
        
        # hand center
        if self._split == "training":
            center_hand = jtr[0]
            jtr = jtr - center_hand
            vert = vert - center_hand
            obj_point = obj_point - center_hand
        else:
            # if self.model == "Obman" :
            center_hand = meta_data["handJoints3D"] * 1000
            # center_hand[0] = -center_hand[0]
            obj_point = obj_point - center_hand

        img_list, mask_list, label2d_list, camera = cut_img(img_list, mask_list, label2d_list, K)
        image_crop = img_list[0]
        mask_crop = mask_list[0]

        obj_point_2d = torch.tensor(xyz2uvd(obj_point, camera)[:,:2])
        center_2d = obj_point_2d.mean(0)
        centered_point = obj_point_2d - center_2d
        scale_2d = centered_point.norm(dim=1).max(0)[0]

        # adding noise like Freihand dataloader
        image_crop = torch.tensor(image_crop).permute(2, 0, 1)
        image_crop = func_transforms.normalize(image_crop / 255, [0.5, 0.5, 0.5], [1, 1, 1])
        mask_crop3 = torch.tensor(mask_crop)
        mask_crop = (mask_crop3 / 255).mean(2).unsqueeze(2).permute(2, 0, 1)
        mask_crop3 = func_transforms.normalize(mask_crop3.permute(2, 0, 1) / 255, [0.5, 0.5, 0.5], [1, 1, 1])

        prior_mask = torch.cat([mask_crop, pro_xz_prior, zeros], dim=0)
        prior_mask = func_transforms.normalize(prior_mask, [0.5, 0.5, 0.5], [1, 1, 1])
    
        gt_mask = torch.cat([mask_crop, pro_xz_gt, zeros], dim=0)
        gt_mask = func_transforms.normalize(gt_mask, [0.5, 0.5, 0.5], [1, 1, 1])

        if self.model == "Obman" or self.model == "TemplateHOI":
            if self._split == 'training':
                return image_crop, mask_crop, prior_mask, \
                    obj_point, torch.tensor(template), vert, torch.squeeze(jtr), \
                    camera, center_2d, scale_2d
            else:
                return image_crop, mask_crop, prior_mask, \
                    obj_point, torch.tensor(template), \
                    camera, center_2d, scale_2d
        elif self.model == "PoseHOI":
            if self._split == "training":
                return image_crop, prior_verts, uni_verts, gt_prior_verts, prior_faces, \
                        obj_point, mask_gt, vert, torch.squeeze(jtr), camera
            else:
                return image_crop, prior_verts, uni_verts, gt_prior_verts, prior_faces, \
                        obj_point, mask_gt, camera
        else:
            return image_crop, prior_mask, gt_mask, obj_point, torch.tensor(template), gt_template, object_rot.T, new_rot.T

if __name__ == '__main__':
    print("start")
    data_train = HO3DDataset(setup = 's0', split ='training', model="TemplateHOI")
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    for step, (image_crop, mask_crop, mix_mask_crop, obj_point_gt, obj_coarse_pc, handverts_gt, handjoint_gt, camera, center_2d, scale_2d) in enumerate(data_loader_train):
        image = ((image_crop.numpy() + 0.5) * 255).astype(np.uint8)
        img_pil = Image.fromarray(image)
        img_pil.save("image_crop")
        break