import torch
from data.data import load_mnist_temporal
from mind.encoders import ImageEncoder, TextEncoder
from mind.reconstructor import Reconstructor
from mind.association import TemporalAssociationLayer
from mind.memory import TemporalMemory
from training.training import train_loop
from config.config import load_config
import os

def main():
    config = load_config("config/config.yaml")

    dataloader, train_indices = load_mnist_temporal(
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
        digits=config["digits"],
        time_stride=config["time_stride"],
        noise_std=config["noise_std"],
        gap_between_digits=config["gap_between_digits"],
        train=True,
        return_indices=True
    )

    import json
    with open("outputs/train_indices.json", "w") as f:
        json.dump(train_indices, f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import os

    # === Encoder inizializzazione ===
    image_encoder = ImageEncoder(embedding_dim=config["embedding_dim"]).to(device)
    text_encoder = TextEncoder(vocab_size=config["vocab_size"], embedding_dim=config["embedding_dim"]).to(device)
    reconstructor = Reconstructor(config["embedding_dim"], output_dim=28*28).to(device)

    # === Percorsi pesi salvati ===
    img_path = "outputs/associative/image_encoder.pth"
    txt_path = "outputs/associative/text_encoder.pth"
    # (il reconstructor lo gestirai in futuro)

    # === Caricamento se esistono i pesi ===
    if os.path.exists(img_path) and os.path.exists(txt_path):
        print("📦 Pesi trovati. Caricamento in corso...")
        image_encoder.load_state_dict(torch.load(img_path, map_location=device))
        text_encoder.load_state_dict(torch.load(txt_path, map_location=device))
    else:
        print("🆕 Nessun peso trovato. Allenamento da zero.")
    assoc_layer = TemporalAssociationLayer(config["tau"]).to(device)
    memory = TemporalMemory()

    params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(reconstructor.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"])

    for epoch in range(config["epochs"]):
        train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']} complete")

    os.makedirs("outputs/associative", exist_ok=True)
    torch.save(image_encoder.state_dict(), "outputs/associative/image_encoder.pth")
    # Salva la memoria (dopo il training)
    print(f"🔍 Numero elementi in memoria: {len(memory.embeddings)}")
    torch.save([emb.cpu() for emb in memory.embeddings], "outputs/associative/memory_embeddings.pth")
    torch.save(memory.times, "outputs/associative/memory_times.pth")
    print("Memoria salvata.")
    torch.save(text_encoder.state_dict(), "outputs/associative/text_encoder.pth")
    print("Training completato. Pesi salvati.")


if __name__ == "__main__":
    main()
