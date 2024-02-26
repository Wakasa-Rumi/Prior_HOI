import torch
import pickle
import pytorch3d.transforms.rotation_conversions as rot_cvt
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import load_objs_as_meshes
from obman_net.mano_train.networks.branches import atlasutils
from preprocess.generate_wrap import visualize
from matplotlib import pyplot as plt
from loss.Render import Render
import math
import numpy as np
import os

class_names = [
    # "010_potted_meat_can",
    # "021_bleach_cleanser",
    "019_pitcher_base",
    "003_cracker_box",
    "006_mustard_bottle",
    "004_sugar_box",
    "035_power_drill",
    "011_banana",
    "037_scissors",
    "025_mug"
]

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

def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat

def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat

def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot
    
def align(prior, template):
    centered_pior = prior - prior.mean(0)[0]
    centered_template = template - template.mean(0)[0]
    scale_prior = torch.norm(centered_pior, p=2, dim=1).max(0)[0]
    scale_template = torch.norm(centered_template, p=2, dim=1).max(0)[0]

    centered_pior = centered_pior / scale_prior * scale_template
    return centered_pior, centered_template

c_loss = atlasutils.ChamferLoss()
def rotate_distance(x, y, z, prior, template):
    x = x / 180 * math.pi
    y = y / 180 * math.pi
    z = z / 180 * math.pi
    rot = axis_angle_t_to_matrix(torch.tensor([x, y, z]))[:3,:3]

    prior = torch.matmul(prior, rot.T)
    c1, c2 = c_loss(prior.unsqueeze(0).cuda(), template.unsqueeze(0).cuda())

    return torch.mean(c1 + c2), rot 

rot_json = {}

def gen_rotated_prior(dataset):
    for class_name in class_names:
        fig = plt.figure(figsize=(12, 12))
        # prior_file = "preprocess/template/{}.pth".format(class_name)
        # pc = torch.load(prior_file)

        prior_mesh_file = "datapreprocess/dexycb1/mesh_obj/{}.obj".format(class_name)
        prior_mesh = load_objs_as_meshes([prior_mesh_file])
        pc = prior_mesh.verts_packed()

        if dataset == "ho3d":
            f = open("/mnt/c/Users/huangyiyao/Desktop/HOI/generate_mask/ho3d_sample/ho3d_vid_train_mesh.pkl", "rb")
            obj_data = pickle.load(f)
            template = obj_data[class_name].verts_packed()
        elif dataset == "dexycb":
            template = np.loadtxt(os.path.join("/home/yiyao/HOI/datasets/dexycb", 'models', class_name, 'points.xyz'))

        # load mesh
        
        prior, template = align(pc, template)

        distance = 9999999
        rot_matrix = torch.zeros(3, 3)
        axis = {"x":0, "y":0, "z":0}
        for x in range(0, 30):
            for y in range(0, 30):
                for z in range(0, 30):
                    dis, rot = rotate_distance(x*12, y*12, z*12, prior, template)
                    if dis < distance:
                        distance = dis
                        rot_matrix = rot
                        axis["x"] = x
                        axis["y"] = y
                        axis["z"] = z
            print("{} x:".format(class_name), x)
        
        rotatated_prior = torch.matmul(prior, rot_matrix.T)
        torch.save(rotatated_prior, "datapreprocess/dexycb1/rot_mesh_obj/{}.pth".format(class_name))

        visualize(fig, rotatated_prior, template, None, None, "datapreprocess/dexycb1/rot_mesh_obj/{}.jpg".format(class_name))
        print(axis)

        rot_json[class_name] = rot_matrix
    fw = open("datapreprocess/dexycb1/rot_mesh_obj/rot_matrix.pth", "wb")
    pickle.dump(rot_json, fw)
    # fw = open("preprocess/rotated_prior/rot_matrix.pth", "rb")
    # print(pickle.load(fw))

def rotate_distance(x, y, z, prior, template):
    x = x / 180 * math.pi
    y = y / 180 * math.pi
    z = z / 180 * math.pi
    rot = axis_angle_t_to_matrix(torch.tensor([x, y, z]))[:3,:3]

    prior = torch.matmul(prior, rot.T)
    c1, c2 = c_loss(prior.unsqueeze(0).cuda(), template.unsqueeze(0).cuda())

    return torch.mean(c1 + c2), rot 

rot_json = {}

def fine_tune(dataset):
    for class_name in class_names:
        fig = plt.figure(figsize=(12, 12))
        prior_file = "datapreprocess/dexycb1/rot_mesh_obj/{}.pth".format(class_name)
        pc = torch.load(prior_file)

        prior_mesh_file = ""

        if dataset == "ho3d":
            f = open("/mnt/c/Users/huangyiyao/Desktop/HOI/generate_mask/ho3d_sample/ho3d_vid_train_mesh.pkl", "rb")
            obj_data = pickle.load(f)
            template = obj_data[class_name].verts_packed()
        elif dataset == "dexycb":
            template = np.loadtxt(os.path.join("/home/yiyao/HOI/datasets/dexycb", 'models', class_name, 'points.xyz'))
        
        prior, template = align(pc, template)

        distance = 9999999
        rot_matrix = torch.zeros(3, 3)
        axis = {"x":0, "y":0, "z":0}
        for x in range(0, 8):
            for y in range(0, 8):
                for z in range(0, 8):
                    dis, rot = rotate_distance(x, y, z, prior, template)
                    # dis, rot, mask1, mask2, mask3 = rotate_render_distance(x, y, z, prior, template)
                    if dis < distance:
                        distance = dis
                        rot_matrix = rot
                        axis["x"] = x
                        axis["y"] = y
                        axis["z"] = z
            print("{} x:".format(class_name), x)
        
        rotatated_prior = torch.matmul(prior, rot_matrix.T)
        torch.save(rotatated_prior, "preprocess/rotated_prior/fine_{}.pth".format(class_name))

        visualize(fig, rotatated_prior, template, None, None, "preprocess/rotated_prior/fine_{}.jpg".format(class_name))
        print(axis)

        rot_json[class_name] = rot_matrix
    fw = open("preprocess/rotated_prior/fine_rot_matrix.pth", "wb")
    pickle.dump(rot_json, fw)
    # fw = open("preprocess/rotated_prior/rot_matrix.pth", "rb")
    # print(pickle.load(fw))