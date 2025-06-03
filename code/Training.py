import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Optional

from Models import ContrastiveLoss, SupConLoss, combined_loss
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
    save_dir: str = "../models",
):
    # Auto device selection
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training")

    # Default loss function if not provided
    if loss_fn is None:
        loss_fn = combined_loss

    # Wrap the dataset into torch Dataset and DataLoader
    torch_dataset = SiameseTorchDataset(dataset, num_pairs=num_pairs)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, label, y1, y2 in dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2, mode='siamese')
            logits1 = model(x1, mode='classify')
            logits2 = model(x2, mode='classify')

            loss = loss_fn(emb1, emb2, logits1, logits2, label, y1, y2)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save trained model
    save_path = save_dir + f"/siamese_model_{dataset.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# ...existing code...

def train_hard_negative_siamese(
    model: nn.Module,
    dataset: "Dataset",
    optimizer: torch.optim.Optimizer,
    embeddings,
    hard_negative_threshold: float = 0.9,
    loss_fn: Optional[nn.Module] = None,
    batch_size: int = 64,
    num_pairs: int = 5000,
    epochs: int = 10,
    device: Optional[str] = None,
    save_dir: str = "../models"
):
    """
    Trains a Siamese network using hard negative mining.
    Args:
        model: SiameseNetwork instance.
        dataset: Dataset object.
        optimizer: Optimizer.
        embeddings: Precomputed embeddings for the dataset.
        hard_negative_threshold: Threshold for selecting hard negatives.
        loss_fn: Loss function.
        batch_size: Batch size.
        num_pairs: Number of pairs to generate.
        epochs: Number of epochs.
        device: Device to use.
        save_dir: Directory to save the model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for hard negative training")

    if loss_fn is None:
        loss_fn = combined_loss

    # Create a SiameseTorchDataset with hard negative mining
    torch_dataset = SiameseTorchDataset(
        base_dataset=dataset,
        num_pairs=num_pairs,
        seed=42,
        embeddings=embeddings,
        hard_negative_threshold=hard_negative_threshold
    )
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, label, y1, y2 in dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2, mode='siamese')
            logits1 = model(x1, mode='classify')
            logits2 = model(x2, mode='classify')

            loss = loss_fn(emb1, emb2, logits1, logits2, label, y1, y2)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Hard Negative] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save trained model
    save_path = save_dir + f"/siamese_model_{dataset.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Hard negative model saved to {save_path}")

def compute_embeddings(model, dataset, device='cpu'):
    model.eval()
    X, _ = dataset.get_processed_data()
    inputs = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        embeddings = model.forward_once(inputs).cpu().numpy()
    return embeddings