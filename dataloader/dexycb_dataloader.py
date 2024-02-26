# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import warnings
import os
import yaml
import numpy as np
#from manopth.manolayer import ManoLayer
import torch
import torchvision
import torch.utils.data as data
# import matplotlib.pyplot as plt
from obman_net.manopth_master.manopth.manolayer import ManoLayer
# import imageio
from PIL import Image
import cv2
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
from torchvision.transforms import functional as func_transforms

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

    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts)
    plt.savefig('figs/atlas/obj{}.png'.format(num))


def show_images_list(batch: torch.Tensor, idx):
    """ Display a batch of images inline. """
    # print(batch.permute(1, 2, 0).numpy().shape)
    # scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray((batch).numpy().astype(np.uint8)).save("/home/yiyao/HOI/hand_object_trans/figs/list/img_list{}.jpg".format(idx))

def show_images(batch: torch.Tensor, idx):
    """ Display a batch of images inline. """
    # print(batch.permute(1, 2, 0).numpy().shape)
    # scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray((batch).numpy().astype(np.uint8)).save("/home/yiyao/HOI/hand_object_trans/figs/crop/img_crop{}.jpg".format(idx))

def get_mask(mask_dir, f):
    # print("label_2d", label_2d[0].shape)
    mask_format = "{:06d}"
    mask_frame = mask_format.format(f)
    mask_list = []
    for idx in range(9):
        path = mask_dir + "/" + mask_frame + "_00000{}".format(idx) + ".png"
        if not os.path.exists(path):
            break
        mask_image = Image.open(path).convert("RGB")
        mask_list.append(mask_image)
    return mask_list

def cut_masks(mask_img_list, label2d_list, radio=0.7, img_size=256):
    vis_ori = []

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

    mask_img_list_out = []
    for m_img in mask_img_list:
        mask_img_list_out.append(cv2.warpAffine(m_img, M, dsize=(img_size, img_size)))
    
    vis = []
    for img in mask_img_list_out:
        vis_num = 0
        for i in range(256):
            for j in range(256):
                if img[i][j][0].item() != 0:
                    vis_num += 1
        vis.append(vis_num)

    obj_num = 0
    for i in range(len(mask_img_list_out)):
        if vis[i] > 500:
            obj_num += 1
    
    return obj_num

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

# _SUBJECTS = [
#     '20200709-subject-01',
# ]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

# _SERIALS = [
#     '836212060125',
# ]

_YCB_CLASSES = {
    1: '002_master_chef_can',  #category 0
    2: '003_cracker_box',      #category 1
    3: '004_sugar_box',        #category 1
    4: '005_tomato_soup_can',  #category 0
    5: '006_mustard_bottle',   #category 0
    6: '007_tuna_fish_can',    #category 0
    7: '008_pudding_box',      #category 1
    8: '009_gelatin_box',      #category 1
    9: '010_potted_meat_can',  #category 1
    10: '011_banana',          #category 0
    11: '019_pitcher_base',    #category 0
    12: '021_bleach_cleanser', #category 0
    13: '024_bowl',            #category 2
    14: '025_mug',             #category 3
    15: '035_power_drill',     #category 4
    16: '036_wood_block',      #category 1
    17: '037_scissors',        #category 6
    18: '040_large_marker',    #category 0
    19: '051_large_clamp',     #category 5
    20: '052_extra_large_clamp', #category 5
    21: '061_foam_brick',      #category 1
}
_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


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
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd


def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256):
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

    return img_list_out, label2d_list_out, camera

def interact(hand_list, obj_list):
    Min_hand = []
    Min_obj = []
    Max_hand = []
    Max_obj = []
    for hand in hand_list:
        Min_hand.append(np.min(hand, axis=0))
        Max_hand.append(np.max(hand, axis=0))
    for obj in obj_list:
        Min_obj.append(np.min(obj, axis=0))
        Max_obj.append(np.max(obj, axis=0))

    Min_hand = np.min(np.array(Min_hand), axis=0)
    Max_hand = np.max(np.array(Max_hand), axis=0)
    Min_obj = np.min(np.array(Min_obj), axis=0)
    Max_obj = np.max(np.array(Max_obj), axis=0)    

    if Min_hand[0] > Max_obj[0] or Max_hand[0] < Min_obj[0] or Min_hand[1] > Max_obj[1] or Max_hand[1] < Min_obj[1]:
        return False
    return True

def rotate(origin, point, angle, scale):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + scale * math.cos(angle) * (px - ox) - scale * math.sin(angle) * (py - oy)
    qy = oy + scale * math.sin(angle) * (px - ox) + scale *  math.cos(angle) * (py - oy)

    return qx, qy

def rgb_processing(rgb_img):
    # in the rgb image we add pixel noise in a channel-wise manner
    noise_factor = 0.4
    pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    return rgb_img


def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])


    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    #result = Minimal(**d)

    return d

def point_normalize(pc): # [n, 3]
    center = pc.mean(0)
    centered_pc = pc - center
    scale = torch.norm(centered_pc, p=2, dim=1).max(0)[0]

    nor_pc = centered_pc / scale
    # center = (pc.max(0)[0] + pc.min(0)[0]) / 2
    # centered_pc = pc - center
    # radius = np.linalg.norm(centered_pc, 2, 1).max()
    # nor_pc = centered_pc / radius

    return nor_pc, centered_pc, scale

class DexYCBDataset(data.Dataset):
    """DexYCB dataset."""
    ycb_classes = _YCB_CLASSES
    mano_joints = _MANO_JOINTS
    mano_joint_connect = _MANO_JOINT_CONNECT

    def __init__(self, setup, split, model):
        """Constructor.

        Args:
          setup: Setup name. 's0', 's1', 's2', or 's3'.
          split: Split name. 'train', 'val', or 'test'.
        """
        self._setup = setup
        self._split = split
        self.model = model

        #assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        self._data_dir = "/home/yiyao/HOI/datasets/dexycb/"
        self._data_dir_pe = "/home/yiyao/HOI/datasets/dexycb/coarse_clouds/"
        self._data_dir_pe_val = "/home/yiyao/HOI/datasets/dexycb/coarse_clouds_val/"
        self._data_dir_pe_test = "/home/yiyao/HOI/datasets/dexycb/coarse_clouds_test/"
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._model_dir = os.path.join(self._data_dir, "models")

        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._h = 480
        self._w = 640

        self._obj_file = {
            k: os.path.join(self._model_dir, v, "textured_simple.obj")
            for k, v in _YCB_CLASSES.items()
        }

        # Seen subjects, camera views, grasped objects.
        #这里其实进行了数据划分了，因为!=4这里，所以训练集，测试集都可以拿出来了
        if self._setup == 's0':
            if self._split == 'training':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                # subject_ind = [0]
                # serial_ind = [0]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if self._split == 'val':
                subject_ind = [0, 1]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                # subject_ind = [0]
                # serial_ind = [0]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if self._split == 'test':
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                # subject_ind = [0]
                # serial_ind = [0]                
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics",
                                     "{}_{}x{}.yml".format(s, self._w, self._h))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        self._sequences = []
        self._mapping = []
        self._ycb_ids = []
        self._ycb_grasp_ind = []
        self._mano_side = []
        self._mano_betas = []
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]

            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials)) #number of serials
                f = np.arange(meta['num_frames']) #num_frames
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
                self._ycb_ids.append(meta['ycb_ids'])
                self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])
                self._mano_side.append(meta['mano_sides'][0])
                mano_calib_file = os.path.join(self._data_dir, "calibration",
                                               "mano_{}".format(meta['mano_calib'][0]),
                                               "mano.yml")
                with open(mano_calib_file, 'r') as f:
                    mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                self._mano_betas.append(mano_calib['betas'])
            offset += len(seq)
        self._mapping = np.vstack(self._mapping)

        self.valid_idx_list = np.load(os.path.join(self._data_dir,'valid_idx_list.npy')).tolist()
        self.idx_list = np.load('hand_pipline/valid_lists/valid_idx_list.npy').tolist()
        self.valid_idx_list_val = np.load(os.path.join(self._data_dir,'valid_idx_list_val.npy')).tolist()
        self.idx_list_val = np.load('hand_pipline/valid_lists/valid_idx_list_val.npy').tolist()
        self.valid_idx_list_test = np.load(os.path.join(self._data_dir,'valid_idx_list_test.npy')).tolist()
        self.idx_list_test = np.load('hand_pipline/valid_lists/valid_idx_list_test.npy').tolist()

        ############################load objects#####################################
        objName_list = os.listdir(os.path.join(self._data_dir, 'models'))
        objmesh_all = {}
        objects_point_all = {}
        for i in range(len(objName_list)):
            objMesh = read_obj(
                os.path.join(self._data_dir, 'models', objName_list[i], 'textured_simple.obj'))
            object_points = np.loadtxt(os.path.join(self._data_dir, 'models', objName_list[i], 'points.xyz'))
            objmesh_all[objName_list[i]] = objMesh
            objects_point_all[objName_list[i]] = object_points
        self.objmesh_all = objmesh_all
        self.objects_point_all = objects_point_all

        self.bop_path_train = '/home/yiyao/HOI/datasets/dexycb/bop/s0/train'
        self.bop_path_val = '/home/yiyao/HOI/datasets/dexycb/bop/s0/val'
        self.bop_path_test = '/home/yiyao/HOI/datasets/dexycb/bop/s0/test'
        self._bop_format = "{:06d}"
        self._fbop_format = "{:06d}.jpg"
        self._mbop_format = "{:06d}_000000.png"

    def __len__(self):
        if self._split=='training':
            # return self.valid_idx_list.shape[0] 
            return len(self.idx_list)
        if self._split=='val':
            # return self.valid_idx_list_val.shape[0]
            return len(self.idx_list_val)
        if self._split=='test':
            # return self.valid_idx_list_test.shape[0]
            return len(self.idx_list_test)
            # return 128

    def __getitem__(self, idx):
        idx = idx
        if self._split=='training':
            s, c, f = self._mapping[self.valid_idx_list[self.idx_list[idx]]]
            d_pe = self._data_dir_pe + "pc_" + str(idx) + ".pth"
            bop_idx = s*8 + c
        if self._split=='val':
            s, c, f = self._mapping[self.valid_idx_list_val[self.idx_list_val[idx]]]
            d_pe = self._data_dir_pe_val + "pc_" + str(idx) + ".pth"
            bop_idx = s*8 + c
        if self._split=='test':
            s, c, f = self._mapping[self.valid_idx_list_test[self.idx_list_test[idx]]]
            d_pe = self._data_dir_pe_test + "pc_" + str(idx) + ".pth"
            bop_idx = s*8 + c

        d = os.path.join(self._data_dir, self._sequences[s], self._serials[c]) # dataset/20200709-subject-01/20200709_142123/836212060125
        sample = {
            'color_file': os.path.join(d, self._color_format.format(f)),
            'depth_file': os.path.join(d, self._depth_format.format(f)),
            'label_file': os.path.join(d, self._label_format.format(f)),
            'intrinsics': self._intrinsics[c],
            'ycb_ids': self._ycb_ids[s],
            'ycb_grasp_ind': self._ycb_grasp_ind[s],
            'mano_side': self._mano_side[s],
            'mano_betas': self._mano_betas[s],
        }
        # Load poses.
        label = np.load(sample['label_file'])
        pose_m = label['pose_m']
        hand_pose = torch.from_numpy(pose_m)
        joint_3d = label['joint_3d'][0]
        joint_2d = label['joint_2d'][0] 

        #####load object ################
        label = np.load(sample['label_file'])
        object_name_index = sample['ycb_ids'][sample['ycb_grasp_ind']]

        object_name = self.ycb_classes[object_name_index]
        if self.model != "PoseBlock":
            d_pe = "/home/yiyao/HOI/datasets/dexycb/pe_template/" + object_name + ".pth"
        else:
            d_pe = "template/" + object_name + ".pth"
        obj_coarse_pc = torch.tensor(torch.load(d_pe)).permute(1, 0) * 1000
        pose_y = label['pose_y']
        
        object_rotation = pose_y[sample['ycb_grasp_ind']][:,:3]
        object_trs = pose_y[sample['ycb_grasp_ind']][:,3]
        objMesh = self.objmesh_all[object_name]

        obj_point_model = self.objects_point_all[object_name][:1024]
        obj_point = (np.matmul(obj_point_model, object_rotation.T) + object_trs) * 1000
        obj_pose = {"object_rotation": object_rotation, "object_trs": object_trs}

        image = Image.open(sample['color_file']).convert("RGB")

        # load Add camera.
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Load MANO layer.
        mano_layer = ManoLayer(flat_hand_mean=False,
                                ncomps=45,
                                side=sample['mano_side'],
                                mano_root='misc/mano',
                                use_pca=True)
        faces = mano_layer.th_faces.numpy()
        betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        #Add MANO meshes.
        vert, jtr = mano_layer(hand_pose[:, 0:48], betas, hand_pose[:, 48:51])
        vert = vert[0]
        jtr = jtr[0]

        # apply current pose to the object model
        if sample['mano_side'] =='left':
            obj_point[:,0] = -obj_point[:,0]
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            vert[:,0] = -vert[:,0]
            jtr[:,0] = -jtr[:,0]
            cx = -(cx-image.size[0])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        obj_point_uvd = xyz2uvd(obj_point, K)
        obj_point_2d = obj_point_uvd[:, :2]

        # center hand
        # if self.model == "Obman":
        center_hand = jtr[0]
        jtr = jtr - center_hand
        vert = vert - center_hand
        obj_point = torch.tensor(obj_point) - torch.tensor(center_hand)

        # crop the image and generate the new K
        image = np.array(image)
        img_list = [image]

        label2d_list = [obj_point_2d, joint_2d]
        img_list, label2d_list, camera = cut_img(img_list, label2d_list, K)

        obj_point_uvd = xyz2uvd(obj_point, camera)
        image_crop = img_list[0]

        if self.model == "GenMask":
            return torch.tensor(image_crop), object_name

        # adding noise like Freihand dataloader
        image_crop = torch.tensor(image_crop).permute(2, 0, 1)
        image_crop = func_transforms.normalize(image_crop / 255, [0.5, 0.5, 0.5], [1, 1, 1])

        obj_name = object_name

        if self.model == "Obman":
            return image_crop, obj_point, vert, torch.squeeze(jtr)

_sets = {}

for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
        name = '{}_{}'.format(setup, split)
        _sets[name] = (lambda setup=setup, split=split: DexYCBDataset(setup, split))


def get_dataset(name):
    """Gets a dataset by name.

    Args:
      name: Dataset name. E.g., 's0_test'.

    Returns:
      A dataset.

    Raises:
      KeyError: If name is not supported.
    """
    if name not in _sets:
        raise KeyError('Unknown dataset name: {}'.format(name))
    return _sets[name]()

inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5])

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # category_train_list_all = [0, 1, 2, 3]
    # valid_train = np.array(category_train_list_all)
    # np.save("figs/valid_idx_list.npy", valid_train)

    Dexycb_train = DexYCBDataset(setup = 's0', split ='val', model='val')
    data_loader_train = torch.utils.data.DataLoader(Dexycb_train,batch_size=16,shuffle=True, num_workers=1)

    # Dexycb_test = DexYCBDataset(setup = 's0', split ='test', model='test')
    # data_loader_test = torch.utils.data.DataLoader(Dexycb_test,batch_size=512,shuffle=False, num_workers=1)
    
    category_train_list_all = []
    for step, (image_crop, obj_point_model, obj_pose, obj_name, vert, jtr) in enumerate(data_loader_train):
        print("image_crop", image_crop.shape) # torch.Size([batch_size, 3, 256, 256])
        print("<obj_point_model>", obj_point_model.shape) # torch.Size([batch_size, 1024, 3])
        print("<object_name>", obj_name) # (batch_size)
        print("<object_rotation>", obj_pose['object_rotation'].shape) # torch.Size([batch_size, 3, 3])
        print("<object_trs>", obj_pose['object_trs'].shape) # torch.Size([batch_size, 3])
        print("<hand_verts>", vert.shape) # torch.Size([batch_size, 778, 3])
        print("<hand_jtrs>", jtr.shape)  # torch.Size([batch_size, 21, 3])
        break