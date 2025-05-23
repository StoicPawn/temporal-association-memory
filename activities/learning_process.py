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
        x_batch, type_batch, time_batch = batch
        x_batch = [x.to(device) for x in x_batch]
        type_batch = type_batch.to(device)
        time_batch = time_batch.to(device)

        loss = 0.0
        count = 0

        for i in range(len(x_batch)):
            x_i = x_batch[i]
            t_i = float(time_batch[i])
            typ = int(type_batch[i].item())

            if typ == 0:  # image
                z_i = image_encoder(x_i.unsqueeze(0)).squeeze(0)
                memory.append(z_i, t_i)

                mem_embs, mem_times = memory.get()
                z_assoc = assoc_layer(mem_embs, mem_times, t_i).detach()  # fisso

                pred = image_encoder(x_i.unsqueeze(0)).squeeze(0)
                loss += association_loss(pred, z_assoc)
                count += 1

            elif typ == 1:  # word
                z_i = text_encoder(x_i.long().unsqueeze(0).to(device)).squeeze(0)
                memory.append(z_i, t_i)

                mem_embs, mem_times = memory.get()
                z_assoc = assoc_layer(mem_embs, mem_times, t_i).detach()  # fisso

                pred = text_encoder(x_i.long().unsqueeze(0).to(device)).squeeze(0)
                loss += association_loss(pred, z_assoc)
                count += 1

            # type == 2 (distractor): ignorato nella loss

        if count > 0:
            loss = loss / count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.4f}")
        else:
            print("Nessun accoppiamento immagine-parola trovato nel batch.")
