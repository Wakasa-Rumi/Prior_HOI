import torch
import numpy as np
import torch.nn.functional as F

def grid_sample(image, optical, mode='zeros'):
    ix = optical[..., 0]
    iy = optical[..., 1]
    if mode == 'zeros':
        image = F.pad(image, (1, 1, 1, 1))
        N, C, IH, IW = image.shape    

        ix = ix * (IW-3) / (IW - 1)
        iy = iy * (IH-3) / (IH - 1)

    N, C, IH, IW = image.shape

    # ix = ((ix + 1) / 2) * (IW-1)
    # iy = ((iy + 1) / 2) * (IH-1)
    ix = ix * (IW-1)
    iy = iy * (IH-1)

    _, H, W, _ = optical.shape

    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def xyz2uvd(query_points, camera):
    fx, fy, fu, fv = camera[:, 0, 0], camera[:, 1, 1], camera[:, 0, 2], camera[:, 1, 2]
    uvd = torch.zeros(query_points.shape).float().cuda()

    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-1)
    fu = fu.unsqueeze(-1)
    fv = fv.unsqueeze(-1)

    uvd[:,:,0] = query_points[:,:,0] * fx / query_points[:,:,2] + fu
    uvd[:,:,1] = query_points[:,:,1] * fy / query_points[:,:,2] + fv
    uvd[:,:,2] = query_points[:,:,2]

    return uvd

def project(query_points, camera, feature_map):
    # query_points: [bs, n, 3]
    # feature_map: [bs, feature_dim, w, h]
    _, n, _ = query_points.shape
    bs, feature_dim, w, h = feature_map.shape

    point_2d = xyz2uvd(query_points, camera)[:,:,:2]
    scale = 256 / w
    point_2d = (point_2d / w / scale).unsqueeze(2) # [bs, N, 1, 2]

    # point_2d = point_2d.unsqueeze(2)

    out_val = grid_sample(feature_map, point_2d).permute(0, 2, 3, 1).squeeze(2)

    return out_val # [bs, N, feature_dim]

def icosphere_2d(sphere_points, mask_center, mask_scare, feature_map):
    # sphere_points [bs, 642, 3]
    # mask_center [bs, 2]
    # mask_scale [bs, 1]
    # feature_map: [bs, feature_dim, w, h]
    bs, feature_dim, w, h = feature_map.shape

    mask_center = mask_center.unsqueeze(1).repeat(1, 642, 1)
    mask_scare = mask_scare.unsqueeze(1).unsqueeze(2)
    point_2d = (sphere_points[:,:,:2] + mask_center) * mask_scare
    scale = 256 / w
    point_2d = (point_2d / scale).unsqueeze(2) # [bs, N, 1, 2]

    out_val = grid_sample(feature_map, point_2d).permute(0, 2, 3, 1).squeeze(2)
    feature_ico = torch.cat([sphere_points, out_val], dim=2)

    return feature_ico