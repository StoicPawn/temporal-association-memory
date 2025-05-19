import torch
import torch.nn.functional as F

def association_loss(z_pred, z_target):
    return F.mse_loss(z_pred, z_target)

def train_loop(image_encoder, text_encoder, reconstructor, assoc_layer, memory, dataloader, optimizer, device):
    image_encoder.train()
    text_encoder.train()
    if reconstructor is not None:
        reconstructor.train()

    for batch in dataloader:
        x_img, x_txt, t_img, t_txt = batch
        x_img = x_img.to(device)
        x_txt = x_txt.to(device)
        t_img = t_img.to(device)
        t_txt = t_txt.to(device)

        z_img = image_encoder(x_img)
        z_txt = text_encoder(x_txt)

        loss = 0.0
        for i in range(len(x_img)):
            memory.append(z_img[i], float(t_img[i]))
            memory.append(z_txt[i], float(t_txt[i]))

            mem_embs, mem_times = memory.get()

            z_assoc_img = assoc_layer(mem_embs, mem_times, float(t_img[i]))
            z_assoc_txt = assoc_layer(mem_embs, mem_times, float(t_txt[i]))

            loss += association_loss(z_assoc_img, z_txt[i]) + association_loss(z_assoc_txt, z_img[i])

        loss /= len(x_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
