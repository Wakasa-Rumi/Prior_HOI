import torch
from dataloader.ho3d_dataloader import HO3DDataset

def generate_gtmask():
    data_train = HO3DDataset
    data_train = HO3DDataset(setup = 's0', split ='training', model="Genmask")
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)