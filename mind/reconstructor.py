import torch.nn as nn

"""ricostruzione finale (opzionale) dall'embedded all'immagine primitiva"""

class Reconstructor(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)