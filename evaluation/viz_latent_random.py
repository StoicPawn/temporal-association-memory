import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mind.encoders import ImageEncoder, TextEncoder
from config.config import load_config
from data.data_random_time import load_mnist_random_time
import matplotlib.cm as cm
import json
import os

def main():
    config = load_config("config/config.yaml")
    embedding_dim = config["embedding_dim"]
    vocab_size = config["vocab_size"]
    digits = config["digits"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modelli ===
    image_encoder = ImageEncoder(input_dim=28*28, embedding_dim=embedding_dim).to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

    image_encoder.load_state_dict(torch.load("outputs/random/image_encoder.pth"))
    text_encoder.load_state_dict(torch.load("outputs/random/text_encoder.pth"))
    image_encoder.eval()
    text_encoder.eval()

    # === Carica train indices se serve
    with open("outputs/random/train_indices.json") as f:
        train_indices = json.load(f)
    forbidden = set()
    for v in train_indices.values():
        forbidden.update(v)

    # === Dataset per visualizzazione
    dataloader = load_mnist_random_time(
        batch_size=1,
        n_samples=config["n_samples"],
        digits=digits,
        noise_std=config["noise_std"],
        train=False,
        return_indices=False
    )

    z_list = []
    labels = []
    types = []

    digit_words = config["digit_words"]
    word_to_idx = {v: k for k, v in digit_words.items()}
    color_map = {digit: cm.get_cmap('tab10')(i % 10) for i, digit in enumerate(digits)}

    with torch.no_grad():
        for x_img, x_txt, t_img, t_txt in dataloader:
            x_img = x_img.to(device)
            x_txt = x_txt.to(device)

            z_img = image_encoder(x_img).squeeze().cpu()
            z_txt = text_encoder(x_txt).squeeze().cpu()

            label_digit = x_txt.item()

            z_list.append(z_img.numpy())
            labels.append(label_digit)
            types.append("image")

            z_list.append(z_txt.numpy())
            labels.append(label_digit)
            types.append("text")

    z_array = np.array(z_list)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, init='random', random_state=42)
    z_2d = tsne.fit_transform(z_array)

    # === Visualizzazione
    plt.figure(figsize=(10, 8))

    for z, label, t in zip(z_2d, labels, types):
        color = color_map[label]
        marker = 'o' if t == "image" else '^'
        alpha = 0.5 if t == "image" else 1.0
        plt.scatter(z[0], z[1], marker=marker, color=color, alpha=alpha)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color_map[d], label=digit_words[d]) for d in digits]
    plt.legend(handles=legend_patches, title="Digit class", loc="best")

    plt.title("Latent Space — Random Temporal Co-Occurrence")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("outputs/random", exist_ok=True)
    plt.savefig("outputs/random/latent_space_random.png")
    print("✅ Spazio latente salvato in outputs/random/latent_space_random.png")
    plt.show()

if __name__ == "__main__":
    main()
