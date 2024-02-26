import torch

def edge_length_loss(pred, gt, face, is_valid=None):

    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    diff1 = torch.abs(d1_out - d1_gt) #* valid_mask_1
    diff2 = torch.abs(d2_out - d2_gt) #* valid_mask_2
    diff3 = torch.abs(d3_out - d3_gt) #* valid_mask_3
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()  