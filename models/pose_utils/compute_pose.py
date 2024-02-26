import torch
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

class_name = "003_cracker_box"

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
    # plt.savefig('visual/obj{}.png'.format(num))
    plt.savefig('{}.png'.format(num))

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu

def get_rot_matrix(rot_tensor):
    bs = rot_tensor.shape[0]
    rot_list = []
    for i in range(bs):
        uu = gram_schmidt(rot_tensor[i]).unsqueeze(0)
        rot_list.append(uu)
    bs_rot_matrix = torch.cat(rot_list, dim=0)
    return bs_rot_matrix

def get_n_vector(b1, b2):
    yz1 = b1[:, 1:].unsqueeze(1)
    yz2 = b2[:, 1:].unsqueeze(1)
    yz = torch.det(torch.cat([yz1, yz2], dim=1)).unsqueeze(1)

    xz1 = b1[:, [0,2]].unsqueeze(1)
    xz2 = b2[:, [0,2]].unsqueeze(1)
    zx = -torch.det(torch.cat([xz1, xz2], dim=1)).unsqueeze(1)

    xy1 = b1[:, :2].unsqueeze(1)
    xy2 = b2[:, :2].unsqueeze(1)
    xy = torch.det(torch.cat([xy1, xy2], dim=1)).unsqueeze(1)

    n_vector = torch.cat([yz, zx, xy], dim=1)
    n_vector = n_vector / torch.norm(n_vector, dim=1).unsqueeze(1)

    return n_vector

def sixpose2rotmatrix(sixpose):
    # sixpose: [bs, 6]
    a1 = sixpose[:, :3]
    a2 = sixpose[:, 3:]
    b1 = a1 / torch.norm(a1, dim=1).unsqueeze(1)
    b2 = a2 - b1 * torch.diag(torch.matmul(b1, a2.transpose(0,1))).unsqueeze(-1)
    b2 = b2 / torch.norm(b2, dim=1).unsqueeze(1)
    b3 = get_n_vector(b1, b2)
    
    b1 = b1.unsqueeze(-1)
    b2 = b2.unsqueeze(-1)
    b3 = b3.unsqueeze(-1)
    rot_matrix = torch.cat([b1, b2, b3], dim=2)
    return rot_matrix

def ninepose2rotmatrix(ninepose):
    # sixpose: [bs, 9]
    a1 = ninepose[:, :3]
    a2 = ninepose[:, 3:6]
    a3 = ninepose[:, 6:9]
    b1 = a1 / torch.norm(a1, dim=1).unsqueeze(1)
    b2 = a2 - b1 * torch.diag(torch.matmul(b1, a2.transpose(0,1))).unsqueeze(-1)
    b2 = b2 / torch.norm(b2, dim=1).unsqueeze(1)
    b3 = a3 - b1 * torch.diag(torch.matmul(b1, a3.transpose(0,1))).unsqueeze(-1) - b2 * torch.diag(torch.matmul(b2, a3.transpose(0,1))).unsqueeze(-1)
    b3 = b3 / torch.norm(b3, dim=1).unsqueeze(1)
    
    b1 = b1.unsqueeze(-1)
    b2 = b2.unsqueeze(-1)
    b3 = b3.unsqueeze(-1)
    rot_matrix = torch.cat([b1, b2, b3], dim=2)
    return rot_matrix

def ninepose2rotmatrix_scale(ninepose):
    # sixpose: [bs, 9]
    a1 = ninepose[:, :3]
    a2 = ninepose[:, 3:6]
    a3 = ninepose[:, 6:9]
    b1 = a1 
    b2 = a2 - b1 * torch.diag(torch.matmul(b1, a2.transpose(0,1))).unsqueeze(-1) / torch.norm(b1, dim=1).unsqueeze(1)
    b3 = a3 - b1 * torch.diag(torch.matmul(b1, a3.transpose(0,1))).unsqueeze(-1) / torch.norm(b1, dim=1).unsqueeze(1) \
            - b2 * torch.diag(torch.matmul(b2, a3.transpose(0,1))).unsqueeze(-1) / torch.norm(b2, dim=1).unsqueeze(1)
    
    b1 = b1.unsqueeze(-1)
    b2 = b2.unsqueeze(-1)
    b3 = b3.unsqueeze(-1)
    rot_matrix = torch.cat([b1, b2, b3], dim=2)
    return rot_matrix

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    # a = torch.randn(2, 3, 3, requires_grad=True)
    # b = gram_schmidt(a)
    # # c = b.sum()
    # # c.backward()
    # # print(b)
    # # print(b.matmul(b.t()))
    # # print(a.grad)
    # pc = torch.load("/home/yiyao/HOI/pointe_text/GLIDE/template/003_cracker_box.pth")
    # display_obj(pc)
    # pc = torch.tensor(pc)
    # # pc = torch.matmul(pc, b.T).detach().numpy()
    # # display_obj(pc)
    # print(a.shape)
    # b = get_rot_matrix(a)
    # print(b.shape)

    # pc_batch = torch.cat([pc.unsqueeze(0), pc.unsqueeze(0)], dim=0)
    # print(pc_batch.shape)
    # r_pc_batch = torch.bmm(pc_batch, b).detach().numpy()

    # display_obj(r_pc_batch[0], num=1)
    # display_obj(r_pc_batch[1], num=2)
    a = torch.randn(4, 6)
    print(a)
    rot_m = sixpose2rotmatrix(a)
    print(rot_m)