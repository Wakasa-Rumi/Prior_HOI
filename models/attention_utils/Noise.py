import torch

def add_gaussian_noise_pc(point_clouds, mean=0, std=1):
    noise = torch.randn(point_clouds.size()).cuda() * std + mean
    noise_pc = point_clouds + noise
    return noise_pc