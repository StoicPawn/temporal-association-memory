import torch
import os
import json
from mind.encoders import ImageEncoder, TextEncoder
from mind.reconstructor import Reconstructor
from mind.association import TemporalAssociationLayer
from mind.memory import TemporalMemory
from training.training import train_loop
from config.config import load_config
from data.data_random_time import load_mnist_random_time  # IMPORTANTE: versione con timestamp casuali

def main():
    config = load_config("config/config.yaml")

    dataloader, train_indices = load_mnist_random_time(
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
        digits=config["digits"],
        noise_std=config["noise_std"],
        train=True,
        return_indices=True
    )

    os.makedirs("outputs/random", exist_ok=True)
    with open("outputs/random/train_indices.json", "w") as f:
        json.dump(train_indices, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_encoder = ImageEncoder(embedding_dim=config["embedding_dim"]).to(device)
    text_encoder = TextEncoder(vocab_size=config["vocab_size"], embedding_dim=config["embedding_dim"]).to(device)
    reconstructor = Reconstructor(config["embedding_dim"], output_dim=28 * 28).to(device)

    assoc_layer = TemporalAssociationLayer(config["tau"]).to(device)
    memory = TemporalMemory()

    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) +
        list(text_encoder.parameters()) +
        list(reconstructor.parameters()),
        lr=config["lr"]
    )

    for epoch in range(config["epochs"]):
        train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device)
        print(f"[Random] Epoch {epoch+1}/{config['epochs']} complete")

    # Salva pesi
    torch.save(image_encoder.state_dict(), "outputs/random/image_encoder.pth")
    torch.save(text_encoder.state_dict(), "outputs/random/text_encoder.pth")
    torch.save([emb.cpu() for emb in memory.embeddings], "outputs/random/memory_embeddings.pth")
    torch.save(memory.times, "outputs/random/memory_times.pth")
    print("[Random] Modello addestrato e salvato.")

if __name__ == "__main__":
    main()
