import os
import torch
import torch.nn as nn
import csv
import json
from sklearn.metrics import confusion_matrix
from config.config import load_config
from mind.encoders import ImageEncoder, TextEncoder
from data.data import load_mnist_temporal

def main():
    # === Config ===
    config = load_config("config/config.yaml")
    digits = config["digits_test_compare"]
    digit_words = {int(k): v for k, v in config["digit_words"].items()}
    word_to_idx = {v: k for k, v in digit_words.items()}
    idx_to_word = {k: v for k, v in digit_words.items()}
    vocab_size = config["vocab_size"]
    embedding_dim = config["embedding_dim"]
    time_stride = config["time_stride"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Carica modelli ===
    image_encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
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
        n_samples=300,
        digits=digits,
        time_stride=time_stride,
        noise_std=0.0,
        gap_between_digits=config["gap_between_digits"],
        train=False,
        forbidden_indices=forbidden,
        return_indices=False
    )

    # === Prepara embedding parole
    word_indices = torch.arange(vocab_size).to(device)
    z_words = text_encoder(word_indices)

    correct = 0
    y_true = []
    y_pred = []
    results = []

    for (x_img, x_txt, _, _) in dataloader:
        img = x_img.to(device)
        true_token_id = x_txt.item()

        # Trova parola e cifra vera
        true_word = None
        for word, idx in word_to_idx.items():
            if idx == true_token_id:
                true_word = word
                break

        true_digit = None
        for d, w in digit_words.items():
            if w == true_word:
                true_digit = d
                break

        with torch.no_grad():
            z_img = image_encoder(img).squeeze()
            dists = torch.norm(z_words - z_img, dim=1)
            pred_token_id = torch.argmin(dists).item()

        pred_word = None
        for word, idx in word_to_idx.items():
            if idx == pred_token_id:
                pred_word = word
                break

        pred_digit = None
        for d, w in digit_words.items():
            if w == pred_word:
                pred_digit = d
                break

        if pred_token_id == true_token_id:
            correct += 1

        y_true.append(true_digit)
        y_pred.append(pred_digit)

        results.append({
            "true_digit": true_digit,
            "true_word": true_word,
            "predicted_digit": pred_digit,
            "predicted_word": pred_word
        })

    acc = 100 * correct / len(results)
    print(f"Associative (model A) accuracy: {acc:.2f}%")

    # === Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=digits)
    print("\nConfusion Matrix:")
    print(cm)

    # === Salva risultati
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/comparison_results_model_A.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_digit", "true_word", "predicted_digit", "predicted_word"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
