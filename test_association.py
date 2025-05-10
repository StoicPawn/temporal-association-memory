# test_association.py

import torch
from torchvision import datasets, transforms
from models.encoders import ImageEncoder, TextEncoder

# Config (coerente con training)
embedding_dim = 64
img_size = 28 * 28
vocab_size = 20
vocab_labels = {3: "three", 6: "six", 9: "nine"}  # semanticamente usate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelli
image_encoder = ImageEncoder(input_dim=img_size, embedding_dim=embedding_dim).to(device)
text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

# Carica pesi
image_encoder.load_state_dict(torch.load("checkpoints/associative/image_encoder.pth"))
text_encoder.load_state_dict(torch.load("checkpoints/associative/text_encoder.pth"))
image_encoder.eval()
text_encoder.eval()

# Carica un'immagine "3" da MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

for img, label in mnist:
    if label == 3:
        img = img.view(-1, 28*28).to(device)
        break

# Encoding immagine
z_img = image_encoder(img)

# Encoding di tutte le 20 parole
word_indices = torch.arange(vocab_size).to(device)
z_words = text_encoder(word_indices)

# Calcolo distanze
dists = torch.norm(z_words - z_img, dim=1)
pred_idx = torch.argmin(dists).item()

# Etichetta predetta
pred_label = vocab_labels.get(pred_idx, f"<unknown:{pred_idx}>")

print(f"Visto numero: 3")
print(f"Rete ha evocato la parola: '{pred_label}' (index: {pred_idx})")

# Stampa Top-5 match
print("\nTop-5 parole evocate:")
topk = torch.topk(-dists, 5).indices.tolist()
for i in topk:
    label = vocab_labels.get(i, f"<unknown:{i}>")
    print(f"  - {label} (index {i}, distanza: {dists[i].item():.4f})")
