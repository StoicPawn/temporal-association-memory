import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_mnist_temporal(batch_size=16, n_samples=1000, digits=None, time_stride=1,
                        noise_std=0.0, gap_between_digits=50, train=True,
                        forbidden_indices=None, return_indices=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_to_idx = {word: idx for idx, word in enumerate(digit_words)}

    x_img, x_txt, t_img, t_txt = [], [], [], []
    indices_per_digit = {}
    current_offset = 0

    if digits is None:
        digits = list(range(10))
    if forbidden_indices is None:
        forbidden_indices = set()

    for digit in digits:
        n_per_digit = n_samples // len(digits)

        # Filtra MNIST: solo immagini del digit e non nei forbidden
        all_indices = [i for i, (_, lbl) in enumerate(mnist)
                       if lbl == digit and i not in forbidden_indices]

        if len(all_indices) < n_per_digit:
            raise ValueError(f"Non abbastanza campioni per la cifra {digit} (richiesti: {n_per_digit}, trovati: {len(all_indices)})")

        # Campiona in modo casuale senza ripetizioni
        chosen_indices = torch.randperm(len(all_indices))[:n_per_digit]
        selected_indices = [all_indices[i] for i in chosen_indices]
        indices_per_digit[digit] = selected_indices

        base_t = float(current_offset)

        for j, idx in enumerate(selected_indices):
            img, label = mnist[idx]
            x_img.append(img)
            x_txt.append(word_to_idx[digit_words[label]])

            delta = 0.02 * j
            t_img.append(base_t + delta)
            t_txt.append(base_t + delta + 0.01)

        current_offset += gap_between_digits * 2

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
