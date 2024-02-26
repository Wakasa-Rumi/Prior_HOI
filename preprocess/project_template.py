import torch
import numpy
import pickle
from PIL import Image

class_name = [
    "010_potted_meat_can",
    "021_bleach_cleanser",
    "019_pitcher_base",
    "003_cracker_box",
    "006_mustard_bottle",
    "004_sugar_box",
    "035_power_drill",
    "011_banana",
    "037_scissors",
    "025_mug"
]

def project_template(template, dim, i_size=256):
    # template: [4096, 3]
    if dim == 0:
        xy = template[:, :2] # xy
    elif dim == 1:
        xy = template[:, 1:] # yz
    elif dim == 2:
        xy = template[:, [0,2]] # xz
    
    scale = torch.norm(xy, p=2, dim=1).max(-1)[0]
    re_xy = (xy / scale * 128) # [4096, 2]
    mask = torch.zeros(i_size * i_size)

    pos_xy = ((re_xy[:, 0] + 128).long() + (re_xy[:, 1] + 128).long() * (i_size))
    mask[pos_xy] = 1
    mask = mask.view(i_size, i_size)

    # for pos in re_xy:
    #     x = int(pos[0] + 128) 
    #     y = int(pos[1] + 128)
    #     mask[y][x] = 1
        # if y - 1 > 0:
        #     mask[y-1][x] = 1
        #     if x - 1 > 0:
        #         mask[y-1][x-1] = 1
        #     if x + 1 < 256:
        #         mask[y-1][x+1] = 1
        # if x - 1 > 0:
        #     mask[y][x-1] = 1
        # if y + 1 < 256:
        #     mask[y+1][x] = 1
        #     if x - 1 > 0:
        #         mask[y+1][x-1] = 1
        #     if x + 1 < 256:
        #         mask[y+1][x+1] = 1
        # if x + 1 < 256:
        #     mask[y][x+1] = 1
    return mask

def save_template_mask(projection, dim, name):
    projection = projection.unsqueeze(-1)
    projection = ((projection.repeat(1, 1, 3)).numpy() * 255).astype(numpy.uint8)
    image = Image.fromarray(projection)
    image.save("img_gt/{}/{}.jpg".format(dim, name))

def save_pkl_models(projction_dict):
    torch.save(projction_dict, "mask_gt/dict.pkl")

def load_pkl_models():
    dic = torch.load("mask_gt/dict.pkl")
    return dic # [256, 256]

def load_gt_models():
    objects_point_all = {}
    f = open("/home/yiyao/HOI/datasets/HO3D_v3/cache/ho3d_vid_train_mesh.pkl", 'rb')
    obj_data = pickle.load(f)
    for i in range(0, 10):
        verts_list = obj_data[class_name[i]].verts_list
        verts = verts_list()[0]
        objects_point_all[class_name[i]] = verts
    return objects_point_all

dic = {}

if __name__ == '__main__':
    # objects_point_all = load_gt_models()
    for name in class_name:
        template = torch.load("/home/yiyao/HOI/HOI/ho/template/{}.pth".format(name))
        # template = objects_point_all[name]
        mask_xy = project_template(template, dim=0)
        save_template_mask(mask_xy, "xy", name)
        mask_yz = project_template(template, dim=1)
        save_template_mask(mask_yz, "yz", name)
        mask_xz = project_template(template, dim=2)
        save_template_mask(mask_xz, "xz", name)

        xyz = {"xy": mask_xy, "yz": mask_yz, "xz": mask_xz}
        dic[name] = xyz
    save_pkl_models(dic)
    # load_pkl_models()
