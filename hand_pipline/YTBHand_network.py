import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch_scatter import scatter_add
from hand_pipline.encoder import resnet18, resnet50
from hand_pipline.utils import utils, mesh_sampling 
import os.path as osp 
import pickle
from typing import Optional

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def Pool(x, trans, dim=1):
    """
    :param x: input feature
    :param trans: upsample matrix
    :param dim: upsample dimension
    :return: upsampled feature
    """
    trans = trans.to(x.device)
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out

def spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation):
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        #print("template_fp")
        mesh = Mesh(filename=template_fp)
        # ds_factors = [3.5, 3.5, 3.5, 3.5]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, ds_factors)
        tmp = {
            'vertices': V,#778 3/ 389 3/195 3/98 3/49 3
            'face': F, #1538 3/761 3/374 3/183 3/86 3 
            'adj': A,
            'down_transform': D,
         'up_transform': U
        }

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        #print(template_fp)
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx])#.to(device)
        for idx in range(len(tmp['face']) - 1)
    ]

    down_transform_list = [
        utils.to_sparse(down_transform)#.to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform)#.to(device)
        for up_transform in tmp['up_transform']
    ]

    return spiral_indices_list, down_transform_list, up_transform_list, tmp

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.to(x.device).view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.to(x.device).reshape(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.relu(self.conv(out))
        return out

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class YTBHand(nn.Module):
    """
    re-implementation of YoutubeHand.
    See https://openaccess.thecvf.com/content_CVPR_2020/papers/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.pdf
    """
    def __init__(self, spiral_indices, up_transform):
        super(YTBHand, self).__init__()
        self.in_channels = 3
        self.out_channels = [64,128,256,512]
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u.size(0) for u in self.up_transform] + [self.up_transform[-1].size(1)]
        self.uv_channel = 21

        backbone, self.latent_size = self.get_backbone("ResNet50")
        self.backbone = Encoder(backbone)

        # 3D decoding
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(self.latent_size[0], self.num_vert[-1] * self.out_channels[-1]))
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layers.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
        self.heads = nn.ModuleList()
        for i in range(len(self.out_channels)):
            self.heads.append(SpiralConv(self.out_channels[i], self.in_channels, self.spiral_indices[i]))

    def get_backbone(self, backbone, pretrained=True):
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = (1000, 2048, 1024, 512, 256)
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        else:
            raise Exception("Not supported", backbone)

        return basenet, latent_channel

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 1

        pred = []
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert[-1], self.out_channels[-1])
            else:
                x = layer(x, self.up_transform[num_features - i])
                pred.append(self.heads[-i](x))
        pred.append(self.heads[0](x))

        return pred

    def forward(self, x):
        z = self.backbone(x)
        pred = self.decoder(z)

        return pred