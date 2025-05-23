import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mind.encoders import ImageEncoder, TextEncoder
from config.config import load_config
from data.data import load_temporal_mixed_dataloader
import matplotlib.cm as cm
import json

def main():
    # === Config ===
    config = load_config("config/config.yaml")
    embedding_dim = config["embedding_dim"]
    vocab_size = config["vocab_size"]
    digits = config["digits"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modelli ===
    image_encoder = ImageEncoder(input_dim=28*28, embedding_dim=embedding_dim).to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    image_encoder.load_state_dict(torch.load("outputs/associative/image_encoder.pth"))
    text_encoder.load_state_dict(torch.load("outputs/associative/text_encoder.pth"))
    image_encoder.eval()
    text_encoder.eval()

    # === Dataloader (test)
    dataloader = load_temporal_mixed_dataloader(
        n_samples=config["n_samples"],
        digits=config["digits"],
        distractor_words=config["distractor_words"],
        distractor_ratio=config["distractor_ratio"],
        noise_std=config["noise_std"],
        batch_size=1,
        train=False
    )

    z_list = []
    labels = []
    types = []

    digit_words = config["digit_words"]
    word_to_idx = {v: k for k, v in digit_words.items()}
    color_map = {digit: cm.get_cmap('tab10')(i % 10) for i, digit in enumerate(digits)}

    with torch.no_grad():
        for x_batch, type_batch, time_batch in dataloader:
            for x, typ, t in zip(x_batch, type_batch, time_batch):
                if typ == 2:
                    continue  # salta i distrattori

                label_digit = None
                if typ == 0:  # image
                    z = image_encoder(x.unsqueeze(0).to(device)).squeeze().cpu()
                    label_digit = predict_label_from_mnist(x)  # oppure salva come None
                    types.append("image")
                elif typ == 1:  # word
                    z = text_encoder(x.long().unsqueeze(0).to(device)).squeeze().cpu()
                    label_digit = x.item()
                    types.append("text")
                else:
                    continue

                if label_digit is not None and label_digit in digits:
                    z_list.append(z.numpy())
                    labels.append(label_digit)

    # === t-SNE
    z_array = np.array(z_list)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, init='random', random_state=42)
    z_2d = tsne.fit_transform(z_array)

    # === Visualizzazione
    plt.figure(figsize=(10, 8))
    for z, label, t in zip(z_2d, labels, types):
        color = color_map[label]
        marker = 'o' if t == "image" else '^'
        alpha = 0.6 if t == "image" else 1.0
        plt.scatter(z[0], z[1], marker=marker, color=color, alpha=alpha)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color_map[d], label=digit_words[d]) for d in digits]
    plt.legend(handles=legend_patches, title="Label reale", loc="best")

    plt.title("Spazio latente 2D (t-SNE) — immagini vs parole")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/latent_space_tsne.png", dpi=300)
    plt.show()

def predict_label_from_mnist(x_img):
    # fallback se vuoi evitare label mancanti per immagini
    return None

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
