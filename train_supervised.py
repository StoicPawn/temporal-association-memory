import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Parametri
target_digits = [3, 6, 9]
digit_to_idx = {3: 0, 6: 1, 9: 2}
input_dim = 28 * 28
batch_size = 64
epochs = 10
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset filtrato
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

images, labels = [], []
for i in range(len(mnist_train)):
    img, label = mnist_train[i]
    if label in target_digits:
        images.append(img)
        labels.append(digit_to_idx[label])  # Rietichettato: 3→0, 6→1, 9→2

X = torch.stack(images)
Y = torch.tensor(labels)

dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

model = SupervisedClassifier(input_dim=input_dim, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        loss = criterion(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

# Salvataggio
os.makedirs("checkpoints/supervised", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/supervised/supervised_model.pth")
print("✅ Modello supervisionato salvato in 'supervised_model.pth'")
