import torch
from data.data import load_mnist_temporal
from models.encoders import ImageEncoder, TextEncoder
from models.reconstructor import Reconstructor
from models.association import TemporalAssociationLayer
from core.memory import TemporalMemory
from core.training import train_loop
from core.config import load_config
import os

def main():
    config = load_config("config.yaml")

    dataloader = load_mnist_temporal(
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
        digits=config["digits"],
        time_stride=config["time_stride"],
        noise_std=config["noise_std"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_encoder = ImageEncoder(embedding_dim=config["embedding_dim"]).to(device)
    text_encoder = TextEncoder(vocab_size=config["vocab_size"], embedding_dim=config["embedding_dim"]).to(device)
    reconstructor = Reconstructor(config["embedding_dim"], output_dim=28*28).to(device)
    assoc_layer = TemporalAssociationLayer(config["tau"]).to(device)
    memory = TemporalMemory()

    params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(reconstructor.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"])

    for epoch in range(config["epochs"]):
        train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']} complete")

    os.makedirs("checkpoints/associative", exist_ok=True)
    torch.save(image_encoder.state_dict(), "checkpoints/associative/image_encoder.pth")
    torch.save(text_encoder.state_dict(), "checkpoints/associative/text_encoder.pth")
    print("Training completato. Pesi salvati.")


if __name__ == "__main__":
    main()
