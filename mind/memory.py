import torch

class TemporalMemory:
    def __init__(self):
        self.embeddings = []
        self.times = []

    def append(self, embedding, t):
        self.embeddings.append(embedding.detach())
        self.times.append(t)

    def get(self):
        return torch.stack(self.embeddings), self.times