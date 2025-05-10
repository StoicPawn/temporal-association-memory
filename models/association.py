import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAssociationLayer(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, memory_embeddings, memory_times, current_time):
        times = torch.tensor(memory_times).to(memory_embeddings.device).float()
        t = torch.tensor([current_time]).to(memory_embeddings.device).float()
        weights = torch.exp(-((t - times)**2) / (self.tau**2))
        weights = weights / weights.sum()
        return torch.sum(weights.unsqueeze(1) * memory_embeddings, dim=0)