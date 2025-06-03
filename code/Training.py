import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Optional

from Models import ContrastiveLoss
from DataManagment import SiameseTorchDataset

def train_siamese(
    model: nn.Module,
    dataset: "Dataset",
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None,
    batch_size: int = 64,
    num_pairs: int = 5000,
    epochs: int = 10,
    device: Optional[str] = None,
    save_dir: str = "../models"
):
    # Auto device selection
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training")

    # Default loss function if not provided
    if loss_fn is None:
        loss_fn = ContrastiveLoss(margin=0.5)

    # Wrap the dataset into torch Dataset and DataLoader
    torch_dataset = SiameseTorchDataset(dataset, num_pairs=num_pairs)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, label in dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2)
            loss = loss_fn(emb1, emb2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save trained model
    save_path = save_dir + f"/siamese_model_{dataset.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
