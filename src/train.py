import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(model, dataset, cfg, device):
    os.makedirs("outputs", exist_ok=True)
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 512),
                        shuffle=True, num_workers=cfg.get("num_workers", 2))
    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                     weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for user, item, rating in loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            pred = model(user, item)
            loss = criterion(pred, rating)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(user)

        train_loss = total_loss / len(dataset)
        scheduler.step()
        val_loss = _validate(model, dataset, device, cfg.get("batch_size", 512))
        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/best_model.pt")
            print(f"  -> Saved checkpoint")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


def _validate(model, dataset, device, batch_size):
    dataset.use_val()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for user, item, rating in loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            total += criterion(model(user, item), rating).item() * len(user)
    dataset.use_train()
    return total / len(dataset)
