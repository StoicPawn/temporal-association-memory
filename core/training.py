import torch
import torch.nn.functional as F

def association_loss(z_pred, z_target):
    return F.mse_loss(z_pred, z_target)

def train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device):
    image_encoder.train()
    text_encoder.train()
    reconstructor.train()

    for batch in dataloader:
        x_img, x_txt, t_img, t_txt = batch
        x_img = x_img.to(device)
        x_txt = x_txt.to(device)

        z_img = image_encoder(x_img)
        z_txt = text_encoder(x_txt)

        # Usa solo il primo elemento del batch per il test MVP
        memory.append(z_img[0], float(t_img[0]))
        memory.append(z_txt[0], float(t_txt[0]))

        mem_embs, mem_times = memory.get()

        z_assoc_img = assoc_layer(mem_embs, mem_times, float(t_img[0]))
        z_assoc_txt = assoc_layer(mem_embs, mem_times, float(t_txt[0]))

        loss = association_loss(z_assoc_img, z_txt[0]) + association_loss(z_assoc_txt, z_img[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")