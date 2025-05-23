import os
import torch
from mind.encoders import ImageEncoder, TextEncoder
from mind.association import TemporalAssociationLayer
from mind.memory import TemporalMemory

def initialize_mind(config, device):
    # === Costruzione moduli ===
    image_encoder = ImageEncoder(embedding_dim=config["embedding_dim"]).to(device)
    text_encoder = TextEncoder(vocab_size=config["vocab_size"], embedding_dim=config["embedding_dim"]).to(device)
    assoc_layer = TemporalAssociationLayer(tau=config["tau"]).to(device)
    memory = TemporalMemory()

    # === Percorsi pesi ===
    img_path = config.get("image_encoder_path", "outputs/associative/image_encoder.pth")
    txt_path = config.get("text_encoder_path", "outputs/associative/text_encoder.pth")

    # === Caricamento pesi se esistono ===
    if os.path.exists(img_path):
        print("Caricamento image_encoder da", img_path)
        image_encoder.load_state_dict(torch.load(img_path, map_location=device))
    else:
        print("image_encoder inizializzato da zero.")

    if os.path.exists(txt_path):
        print("Caricamento text_encoder da", txt_path)
        text_encoder.load_state_dict(torch.load(txt_path, map_location=device))
    else:
        print("text_encoder inizializzato da zero.")

    return image_encoder, text_encoder, assoc_layer, memory
