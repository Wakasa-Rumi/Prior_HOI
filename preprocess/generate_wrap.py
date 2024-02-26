import torch
import trimesh
import numpy as np
import pymesh
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle

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
    ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

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

def add_hand_obj_meshes(
    ax,
    prior_verts=None,
    verts=None,
    faces=None,
    num=0
):
    if num == 0:
        a, b, c = 0, 1, 2
    elif num == 1:
        a, b, c = 1, 2, 0
    elif num == 2:
        a, b, c = 2, 0, 1

    if verts is not None:
        rot_verts = torch.zeros_like(torch.tensor(verts)).numpy()
        rot_verts[:,0] = verts[:,a]
        rot_verts[:,1] = verts[:,b]
        rot_verts[:,2] = verts[:,c]

        # visualize_joints_3d(ax, gt_batchjoints3d, joint_idxs=joint_idxs)
        # Add mano predictions
        if faces is not None:
            add_mesh(ax, rot_verts, faces) 
        else:
            ax.scatter(
                rot_verts[:, 0],
                rot_verts[:, 1],
                rot_verts[:, 2],
                c="b",
                alpha=0.2,
                s=5,
            )        

    # Add object gt/predictions
    if prior_verts is not None:
        ax.scatter(
            prior_verts[:, a],
            prior_verts[:, b],
            prior_verts[:, c],
            c="r",
            alpha=0.2,
            s=5,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def visualize(
    fig,
    prior_verts,
    smesh_verts,
    verts,
    faces,
    path
):
    if prior_verts is not None:
        prior_verts = prior_verts.cpu().detach().numpy()
    if verts is not None:
        verts = verts.cpu().detach().numpy()

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    add_hand_obj_meshes(
        ax,
        prior_verts=prior_verts,
        verts=verts,
        faces=faces,
        num=0
    )

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    add_hand_obj_meshes(
        ax,
        prior_verts=prior_verts,
        verts=verts,
        faces=faces,
        num=1
    )

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    add_hand_obj_meshes(
        ax,
        prior_verts=prior_verts,
        verts=verts,
        faces=faces,
        num=2
    )
    plt.savefig(path, dpi=100)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    add_hand_obj_meshes(
        ax,
        prior_verts=smesh_verts,
        verts=None,
        faces=None,
        num=0
    )
    plt.savefig(path, dpi=100)

def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def wrap(verts, prior, delta=0.04, r=0.008, num=25):
    N = verts.shape[0]
    move_vector = torch.zeros_like(verts) - verts

    for _ in range(num):
        P, _ = batch_pairwise_dist(verts.unsqueeze(0), prior.unsqueeze(0)).squeeze(0).min(1)
        # _, idx = batch_pairwise_dist(verts.unsqueeze(0), prior.unsqueeze(0)).squeeze(0).min(0)
        # base_idx = torch.arange(0, N)
        # count = torch.bincount(torch.cat([idx, base_idx], dim=0))
        del_vec = torch.zeros(N, 1)
        del_vec[(P > r)] = delta
        verts = verts + move_vector * del_vec

    return verts

def filter(prior_verts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(prior_verts)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=5.0)
    print(cl)
    return np.asarray(cl.points)

def sample_from_mesh(path):
    sample_num = 1024
    mesh = o3d.io.read_triangle_mesh(path)
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)
    return np.asarray(pcd.points), np.asarray(mesh.vertices), mesh.triangles

def generate_prior_wrap(ico_verts):
    ico_verts = ico_verts
    prior = torch.load("/mnt/c/Users/huangyiyao/Desktop/HOI/pointe_text/GLIDE/template/{}.pth".format(name))
    prior = torch.Tensor(prior)
    prior_centered = prior - prior.mean(0).unsqueeze(0)
    prior_scale = torch.norm(prior_centered, p=2, dim=1).max(0)[0]
    prior = prior_centered / prior_scale

    # ico_verts = wrap(ico_verts, prior)

    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(ico_verts)
    # mesh.triangles = o3d.utility.Vector3iVector(ico_faces)
    # o3d.io.write_triangle_mesh("GLIDE/meshes/{}.obj".format(name), mesh)

    prior = torch.tensor(filter(prior_verts=prior)).float()
    path = "GLIDE/meshes/{}.obj".format(name)
    smesh_pc, ico_verts, ico_faces = sample_from_mesh(path)
    smesh_pc = torch.tensor(smesh_pc).float()
    ico_verts = torch.tensor(ico_verts).float()

    visualize(fig=fig, prior_verts=prior, smesh_verts=smesh_pc, verts=ico_verts, faces=ico_faces,
                path="GLIDE/meshes/images/{}.jpg".format(name))

def generate_template_wrap(class_name, ico_verts, ico_faces):
    f = open("/mnt/c/Users/huangyiyao/Desktop/HOI/generate_mask/ho3d_sample/ho3d_vid_train_mesh.pkl", "rb")
    obj_data = pickle.load(f)
    template = obj_data[class_name].verts_packed()

    print(template.shape)
    trans = template.mean(0)[0]
    centered_template = template - trans
    scale = torch.norm(centered_template, p=2, dim=1).max(0)[0]
    scaled_temp = centered_template / scale

    ico_verts = wrap(ico_verts, scaled_temp)

    ico_verts = ico_verts * scale + trans

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(ico_verts)
    mesh.triangles = o3d.utility.Vector3iVector(ico_faces)
    o3d.io.write_triangle_mesh("GLIDE/meshes/template_{}.obj".format(name), mesh)

    template = torch.tensor(filter(prior_verts=template)).float()
    path = "GLIDE/meshes/template_{}.obj".format(name)
    smesh_pc, ico_verts, ico_faces = sample_from_mesh(path)
    smesh_pc = torch.tensor(smesh_pc).float()
    ico_verts = torch.tensor(ico_verts).float()

    visualize(fig=fig, prior_verts=template, smesh_verts=smesh_pc, verts=ico_verts, faces=ico_faces,
                path="GLIDE/meshes/images/template_{}.jpg".format(name))

if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 12))

    for name in class_name:
        print(name)
        icosphere = trimesh.creation.icosphere(
            subdivisions=3
        )
        ico_faces = np.array(icosphere.faces)
        ico_verts = icosphere.vertices
        ico_verts = torch.Tensor(
            np.array(ico_verts).astype(np.float32)
        )

        generate_template_wrap(name, ico_verts, ico_faces)