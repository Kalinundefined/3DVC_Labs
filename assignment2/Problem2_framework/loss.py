import torch
from torch import nn


class CDLoss(nn.Module):
    """
    CD Loss.
    """

    def __init__(self):
        super(CDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        dist = torch.cdist(prediction, ground_truth, p=2.)
        dist1 = dist.min(dim=0).values.mean()
        dist2 = dist.min(dim=1).values.mean()
        return dist1 + dist2
        # TODO: Implement CD Loss
        # Example:
        #     cd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return cd_loss


class HDLoss(nn.Module):
    """
    HD Loss.
    """
    
    def __init__(self):
        super(HDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        dist = torch.cdist(prediction, ground_truth, p=2.)
        dist1 = dist.min(dim=0).values.max()
        dist2 = dist.min(dim=1).values.max()
        return max(dist1, dist2)
        # TODO: Implement HD Loss
        # Example:
        #     hd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return hd_loss
