import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.encoders import ImageEncoder, TextEncoder
from models.association import TemporalAssociationLayer
from core.memory import TemporalMemory
from core.config import load_config
import os
import csv

# Carica config
config = load_config("config.yaml")
digits = config["digits"]
time_stride = config["time_stride"]
noise_std = config["noise_std"]
embedding_dim = config["embedding_dim"]

digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
word_to_idx = {w: i for i, w in enumerate(digit_words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset filtrato
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

images, labels = [], []
for i in range(len(mnist_test)):
    img, label = mnist_test[i]
    if label in digits:
        images.append(img)
        labels.append(label)

images = torch.stack(images).to(device)
labels = torch.tensor(labels).to(device)

# Applica rumore se richiesto
if noise_std > 0:
    images += torch.randn_like(images) * noise_std
    images = torch.clamp(images, 0.0, 1.0)

# Timestamps
t_img = torch.arange(len(labels)) * time_stride
t_txt = t_img + 1

# Modello supervisionato
class SupervisedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

supervised_model = SupervisedClassifier(28*28, len(digits)).to(device)
supervised_model.load_state_dict(torch.load("checkpoints/supervised/supervised_model.pth"))
supervised_model.eval()

# Modello associativo
image_encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
text_encoder = TextEncoder(vocab_size=20, embedding_dim=embedding_dim).to(device)
assoc_layer = TemporalAssociationLayer(config["tau"]).to(device)
memory = TemporalMemory()

image_encoder.load_state_dict(torch.load("checkpoints/associative/image_encoder.pth"))
text_encoder.load_state_dict(torch.load("checkpoints/associative/text_encoder.pth"))
image_encoder.eval()
text_encoder.eval()

# Costruzione memoria (test)
with torch.no_grad():
    for i in range(len(labels)):
        label = labels[i].item()
        word_idx = word_to_idx[digit_words[label]]
        emb = text_encoder(torch.tensor([word_idx]).to(device))
        memory.append(emb.squeeze(), t=i * time_stride)

# Test comparativo
total = len(labels)
correct_supervised = 0
correct_associative = 0
results = []

with torch.no_grad():
    for i in range(total):
        img = images[i].unsqueeze(0)
        label = labels[i].item()
        gt_word = digit_words[label]

        # Supervised
        logits = supervised_model(img)
        pred_idx = torch.argmax(logits).item()
        pred_digit = digits[pred_idx]
        supervised_word = digit_words[pred_digit]
        if pred_digit == label:
            correct_supervised += 1

        # Associative
        img_emb = image_encoder(img)
        assoc_emb = assoc_layer(memory.get()[0], memory.get()[1], i * time_stride)
        distances = torch.stack([
            torch.norm(assoc_emb - text_encoder(torch.tensor([j]).to(device)).squeeze())
            for j in range(10)
        ])
        best = torch.argmin(distances).item()
        assoc_word = idx_to_word[best]
        if assoc_word == gt_word:
            correct_associative += 1

        results.append({
            "true_digit": label,
            "true_word": gt_word,
            "supervised_word": supervised_word,
            "associative_word": assoc_word
        })

# Salva risultati
os.makedirs("outputs", exist_ok=True)
with open("outputs/comparison_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["true_digit", "true_word", "supervised_word", "associative_word"])
    writer.writeheader()
    writer.writerows(results)

# Print finale
print(f"\nRisultati su {total} esempi:")
print(f"Supervised accuracy:  {100 * correct_supervised / total:.2f}%")
print(f"Associative accuracy: {100 * correct_associative / total:.2f}%")
print("Dati salvati in 'outputs/comparison_results.csv'")
