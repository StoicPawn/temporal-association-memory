import torch
from models.encoders import ImageEncoder, TextEncoder
from models.reconstructor import Reconstructor
from models.association import TemporalAssociationLayer
from core.memory import TemporalMemory

def build_models(embedding_dim, vocab_size, tau, device):
    image_encoder = ImageEncoder(input_dim=28*28, embedding_dim=embedding_dim).to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    reconstructor = Reconstructor(embedding_dim, output_dim=28*28).to(device)
    assoc_layer = TemporalAssociationLayer(tau).to(device)
    memory = TemporalMemory()
    
    params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(reconstructor.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    return image_encoder, text_encoder, reconstructor, assoc_layer, memory, optimizer