import torch
import warnings
import numpy as np
import pickle
from PIL import Image

from dataloader.datasets import get_dataset
from loss.Render import Render
from pytorch3d.structures.meshes import Meshes

if __name__ == '__main__':
    device = torch.device("cuda:0")
    warnings.filterwarnings("ignore")
    print("start")
    data_train, data_val, data_test = get_dataset(dataset_name="ho3d", model_name="GenMask")
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # for step, (image_crop, mask_crop, mix_mask_crop, obj_point_gt, obj_coarse_pc, handverts_gt, handjoint_gt, camera, center_2d, scale_2d) in enumerate(data_loader_train):
    #     image = ((image_crop.squeeze(0).permute(1, 2, 0).numpy() + 0.5) * 255).astype(np.uint8)
    #     img_pil = Image.fromarray(image)
    #     img_pil.save("image_crop.jpg")
    #     break

    f = open("/home/yiyao/HOI/datasets/HO3D_v3/cache/ho3d_vid_train_mesh.pkl", 'rb')
    obj_data = pickle.load(f)

    render = Render()

    # gen_mask
    for step, (obj_verts, obj_faces, gt_mask_dir) in enumerate(data_loader_test):
        meshes = Meshes(obj_verts.cuda(), obj_faces.cuda())

        s = render.rend(meshes) # [bs, 128, 128, 3]

        for i in range(len(gt_mask_dir)):
            mask = s[i]
            mask_img = mask.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
            mask_pil.save(gt_mask_dir[i])
        print(gt_mask_dir[0])