"""
Loss functions.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        sim_matrix = torch.exp(torch.mm(features, features.T) / self.temperature)
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        neg_mask = ~label_mask
        sim_matrix = sim_matrix * (1 - torch.eye(B, device=features.device))
        pos_sim = sim_matrix * label_mask
        denom = sim_matrix.sum(dim=1, keepdim=True)
        loss = -torch.log(pos_sim.sum(dim=1) / denom.squeeze())
        return loss.mean()