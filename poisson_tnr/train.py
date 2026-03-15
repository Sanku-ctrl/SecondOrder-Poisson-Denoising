import torch


def stagewise_train(model, dataloader, optimizer, criterion, device, stage_idx):
    # Freeze all parameters except the stage currently being trained
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Always keep proximal and global parameters trainable if present
    for p in model.prox.parameters():
        p.requires_grad = True
    if hasattr(model, 'data_weight'):
        model.data_weight.requires_grad = True
    if hasattr(model, 'alpha'):
        model.alpha.requires_grad = True
    # enable only current stage
    if stage_idx >= 0 and stage_idx < len(model.stages):
        for p in model.stages[stage_idx].parameters():
            p.requires_grad = True

    model.train()
    total_loss = 0.0
    count = 0
    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        output = model(noisy, noisy, training_stage=stage_idx)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)
    return total_loss / max(count, 1)
