import torch
import torch.nn as nn

"""parte del cervello che trasforma gli oggetti di diversa provenienza in concetti, ovvero li porta nello stesso linguaggio di
oggetti embedded 64"""

class ImageEncoder(nn.Module):
    def __init__(self, input_dim=784, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)