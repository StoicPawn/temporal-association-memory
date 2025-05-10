import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_mnist_temporal(batch_size=16, n_samples=1000, digits=None, time_stride=1, noise_std=0.0, gap_between_digits=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_to_idx = {word: idx for idx, word in enumerate(digit_words)}

    x_img, x_txt, t_img, t_txt = [], [], [], []

    count = 0
    current_offset = 0

    if digits is None:
        digits = list(range(10))

    for digit in digits:
        digit_count = 0
        for i in range(len(mnist)):
            img, label = mnist[i]
            if label != digit:
                continue

            # Add image and word with timestamp
            x_img.append(img)
            x_txt.append(word_to_idx[digit_words[label]])

            base_t = float(digit * 1.0)  # ogni digit inizia da un punto intero diverso
            delta = 0.01 * digit_count   # incremento interno (ravvicinato)

            t_img.append(base_t + delta)
            t_txt.append(base_t + delta + 0.01)

            digit_count += 1
            if digit_count >= n_samples // len(digits):
                break

        # Shift time for the next digit group
        current_offset += gap_between_digits
        count += digit_count

    x_img = torch.stack(x_img)
    x_txt = torch.tensor(x_txt)
    t_img = torch.tensor(t_img)
    t_txt = torch.tensor(t_txt)

    # Optional noise
    if noise_std > 0:
        x_img += torch.randn_like(x_img) * noise_std

    dataset = TensorDataset(x_img, x_txt, t_img, t_txt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
