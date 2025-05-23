import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random

def load_temporal_mixed_dataloader(
    n_samples=1000,
    digits=None,
    distractor_words=None,
    distractor_ratio=0.1,
    gap_range=(0.5, 2.0),
    time_jitter=0.02,
    noise_std=0.0,
    batch_size=16,
    return_indices=False,
    train=True
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])
    mnist = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_to_idx = {word: idx for idx, word in enumerate(digit_words)}

    if digits is None:
        digits = list(range(10))
    if distractor_words is None:
        distractor_words = ["cloud", "tree", "ghost", "lamp", "book"]

    n_digits = len(digits)
    n_per_digit = n_samples // n_digits

    true_events = []
    indices_per_digit = {}
    t_global = 0.0

    for digit in digits:
        all_indices = [i for i, (_, lbl) in enumerate(mnist) if lbl == digit]
        chosen_indices = torch.randperm(len(all_indices))[:n_per_digit]
        selected_indices = [all_indices[i.item()] for i in chosen_indices]
        indices_per_digit[digit] = selected_indices

        for j, idx in enumerate(selected_indices):
            img_tensor, label = mnist[idx]
            if noise_std > 0:
                img_tensor += torch.randn_like(img_tensor) * noise_std

            word_index = word_to_idx[digit_words[label]]

            ε = random.uniform(0, time_jitter)
            t_img = t_global + ε
            t_txt = t_img + 0.01 + random.uniform(0, time_jitter)

            true_events.append((img_tensor, 0, t_img))      # 0 = image
            true_events.append((torch.tensor(word_index), 1, t_txt))  # 1 = word

            t_global += random.uniform(*gap_range)

    # Distractors
    n_distractors = int(len(true_events) * distractor_ratio)
    distractor_events = []
    for _ in range(n_distractors):
        word = random.choice(distractor_words)
        word_index = len(word_to_idx) + distractor_words.index(word)
        t_distr = random.uniform(0, t_global)
        distractor_events.append((torch.tensor(word_index), 2, t_distr))  # 2 = distractor

    # Merge and sort
    all_events = true_events + distractor_events
    all_events.sort(key=lambda x: x[2])  # sort by time

    data_tensors = []
    type_ids = []
    timestamps = []

    for data, type_id, timestamp in all_events:
        data_tensors.append(data if data.dim() > 0 else data.unsqueeze(0))
        type_ids.append(type_id)
        timestamps.append(timestamp)

        # Dataset come lista di tuple (x, type, time)
    events = list(zip(data_tensors, type_ids, timestamps))

    # Define custom collate to return lists
    def custom_collate(batch):
        x_batch, type_batch, time_batch = zip(*batch)
        x_batch = list(x_batch)
        type_batch = torch.tensor(type_batch)
        time_batch = torch.tensor(time_batch)
        return x_batch, type_batch, time_batch

    dataloader = DataLoader(events, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


    if return_indices:
        return dataloader, indices_per_digit
    else:
        return dataloader
