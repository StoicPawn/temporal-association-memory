
import torch
import json
from collections import defaultdict
from mind.encoders import ImageEncoder, TextEncoder
from config.config import load_config
from data.data import load_mnist_temporal
import numpy as np

def main():
    # === Config ===
    config = load_config("config/config.yaml")
    digits = config["digits_test_association"]
    vocab_size = config["vocab_size"]
    embedding_dim = config["embedding_dim"]
    digit_words = {int(k): v for k, v in config["digit_words"].items()}
    word_to_idx = {v: k for k, v in digit_words.items()}
    idx_to_word = {k: v for k, v in digit_words.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Inizializza modelli ===
    image_encoder = ImageEncoder(input_dim=28*28, embedding_dim=embedding_dim).to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

    image_encoder.load_state_dict(torch.load("outputs/associative/image_encoder.pth"))
    text_encoder.load_state_dict(torch.load("outputs/associative/text_encoder.pth"))
    image_encoder.eval()
    text_encoder.eval()

    # === Carica forbidden set
    with open("outputs/train_indices.json") as f:
        train_indices = json.load(f)
    forbidden = set()
    for v in train_indices.values():
        forbidden.update(v)

    # === Carica dati di test
    dataloader = load_mnist_temporal(
        batch_size=1,
        n_samples=len(digits),
        digits=digits,
        time_stride=config["time_stride"],
        noise_std=0.0,
        gap_between_digits=config["gap_between_digits"],
        train=False,
        forbidden_indices=forbidden,
        return_indices=False
    )

    # === Costruisci z_txt per tutto il vocabolario
    word_indices = torch.arange(vocab_size).to(device)
    z_words = text_encoder(word_indices)

    # === Test: confronta z_img con tutti z_words
    print("\n=== TEST A: Associazione senza tempo, confronto diretto nel latente ===")
    for (x_img, x_txt, _, _) in dataloader:
        img_tensor = x_img.to(device)
        target_idx = x_txt.item()
        target_word = idx_to_word[target_idx]

        with torch.no_grad():
            z_img = image_encoder(img_tensor).squeeze()

            dists = torch.norm(z_words - z_img, dim=1)
            sorted_pred = torch.argsort(dists)

            print(f"🖼️ Visto numero: {target_word}")
            print(f"🧠 Primo evocato: {idx_to_word.get(sorted_pred[0].item(), '?')} (distanza: {dists[sorted_pred[0]].item():.4f})")
            print("🎯 Top-3 evocati:")
            for i in range(3):
                label = idx_to_word.get(sorted_pred[i].item(), f"<unknown:{sorted_pred[i].item()}>")
                print(f"  - {label} (distanza: {dists[sorted_pred[i]].item():.4f})")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
