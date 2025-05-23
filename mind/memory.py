import torch

class TemporalMemory:
    def __init__(self):
        self.embeddings = []
        self.times = []

    def append(self, embedding, t):
        if embedding.dim() == 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        self.embeddings.append(embedding.detach())  # passiva
        self.times.append(t)

    def get(self):
        return torch.stack(self.embeddings), self.times
