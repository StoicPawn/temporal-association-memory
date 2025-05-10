import os
import torch
from core.training import train_loop
from data.data import load_mnist_temporal
from init.init import build_models

def train_associative(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = load_mnist_temporal(batch_size=config["batch_size"], n_samples=config["n_samples"])

    image_encoder, text_encoder, reconstructor, assoc_layer, memory, optimizer = build_models(
        embedding_dim=config["embedding_dim"],
        vocab_size=config["vocab_size"],
        tau=config["tau"],
        device=device
    )

    for epoch in range(config["epochs"]):
        train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']} complete")

    os.makedirs("checkpoints/associative", exist_ok=True)

    torch.save(image_encoder.state_dict(), "checkpoints/associative/image_encoder.pth")
    torch.save(text_encoder.state_dict(), "checkpoints/associative/text_encoder.pth")
    print("Training complete. Pesi salvati.")
