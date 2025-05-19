import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random

def load_mnist_random_time(batch_size=16, n_samples=1000, digits=None,
                           noise_std=0.0, train=True, return_indices=False):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_to_idx = {word: idx for idx, word in enumerate(digit_words)}

    x_img, x_txt, t_img, t_txt = [], [], [], []
    indices_per_digit = {}
    all_entries = []

    if digits is None:
        digits = list(range(10))

    for digit in digits:
        n_per_digit = n_samples // len(digits)
        all_indices = [i for i, (_, lbl) in enumerate(mnist) if lbl == digit]
        chosen_indices = torch.randperm(len(all_indices))[:n_per_digit]
        selected_indices = [all_indices[i] for i in chosen_indices]
        indices_per_digit[digit] = selected_indices

        for idx in selected_indices:
            img, label = mnist[idx]
            word = digit_words[label]
            x_img.append(img)
            x_txt.append(word_to_idx[word])
            all_entries.append(len(x_img)-1)  # posizione da randomizzare dopo

    # Assegna timestamp casuali
    random_times = torch.rand(len(x_img)) * 100  # range 0-100
    random.shuffle(all_entries)

    t_img = [random_times[i].item() for i in all_entries]
    t_txt = [random_times[i].item() + 0.01 for i in all_entries]  # still image before word

    x_img = torch.stack(x_img)
    x_txt = torch.tensor(x_txt)
    t_img = torch.tensor(t_img)
    t_txt = torch.tensor(t_txt)

    if noise_std > 0:
        x_img += torch.randn_like(x_img) * noise_std

    dataset = TensorDataset(x_img, x_txt, t_img, t_txt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if return_indices:
        return dataloader, indices_per_digit
    else:
        return dataloader
