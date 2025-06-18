import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Optional

from Models import ContrastiveLoss, SupConLoss, combined_loss
from DataManagment import SiameseTorchDataset, Dataset


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

    train_losses = []
    val_losses = []
    test_losses = []

    # Default loss function if not provided
    if loss_fn is None:
        loss_fn = ContrastiveLoss(margin=0.5)

    # Split dataset into train/val/test
    X, y = dataset.get_processed_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Create torch datasets
    train_ds = SiameseTorchDataset(
        base_dataset=Dataset(dataset.name, X_train, y_train, dataset.feature_names, dataset.target_names),
        num_pairs=num_pairs, seed=42
    )
    val_ds = SiameseTorchDataset(
        base_dataset=Dataset(dataset.name, X_val, y_val, dataset.feature_names, dataset.target_names),
        num_pairs=int(num_pairs * 0.2), seed=43
    )
    test_ds = SiameseTorchDataset(
        base_dataset=Dataset(dataset.name, X_test, y_test, dataset.feature_names, dataset.target_names),
        num_pairs=int(num_pairs * 0.2), seed=44
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.to(device)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        for x1, x2, label, *_ in train_loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2)
            loss = loss_fn(emb1, emb2, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x1, x2, label, *_ in val_loader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                emb1, emb2 = model(x1, x2)
                loss = loss_fn(emb1, emb2, label)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # --- Test ---
        total_test_loss = 0.0
        with torch.no_grad():
            for x1, x2, label, *_ in test_loader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                emb1, emb2 = model(x1, x2)
                loss = loss_fn(emb1, emb2, label)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Save trained model
    save_path = save_dir + f"/siamese_model_{dataset.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # --- Plot Validation vs Test Loss ---
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation vs Train Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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