
import os
import torch
from mind.initializer import initialize_mind
from activities.learning_process import train_loop
from data.data import load_temporal_mixed_dataloader
from config.config import load_config

def run_training(config, device):
    
    # === Inizializza componenti ===
    image_encoder, text_encoder, assoc_layer, memory = initialize_mind(config, device)

    # === Caricamento indici già usati (se esistono) ===
    indices_file = config.get("used_indices_path", "outputs/associative/used_indices.pt")
    if os.path.exists(indices_file):
        used_indices = torch.load(indices_file)
    else:
        used_indices = set()

    # === Creazione DataLoader con filtro su indici ===
    dataloader, new_indices = load_temporal_mixed_dataloader(
        n_samples=config["n_samples"],
        digits=config["digits"],
        distractor_words=config["distractor_words"],
        distractor_ratio=config["distractor_ratio"],
        batch_size=config["batch_size"],
        return_indices=True,
        train=True
    )

    # === Rimuove dati già visti (solo immagini) ===
    filtered_indices = {
        d: [i for i in idxs if i not in used_indices] for d, idxs in new_indices.items()
    }
    all_new_indices = set(i for idxs in filtered_indices.values() for i in idxs)

    # === Aggiorna set indici usati ===
    used_indices.update(all_new_indices)
    os.makedirs("outputs/associative", exist_ok=True)
    torch.save(used_indices, indices_file)

    # === Ottimizzatore ===
    params = list(image_encoder.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"])

    # === Training ===
    for epoch in range(config["epochs"]):
        train_loop(image_encoder, text_encoder, None, assoc_layer, memory, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']} complete")

    # === Salvataggio pesi e memoria ===
    torch.save(image_encoder.state_dict(), config.get("image_encoder_path", "outputs/associative/image_encoder.pth"))
    torch.save(text_encoder.state_dict(), config.get("text_encoder_path", "outputs/associative/text_encoder.pth"))
    torch.save([emb.cpu() for emb in memory.embeddings], "outputs/associative/memory_embeddings.pth")
    torch.save(memory.times, "outputs/associative/memory_times.pth")
    print(f"🔍 Numero elementi in memoria: {len(memory.embeddings)}")
    print("✅ Training completato.")

# ===== Script eseguibile =====

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("config/config.yaml")

    run_training(config, device)
