from DataManagment import DistDataLoader, SiameseTorchDataset
from Models import SiameseNetwork
from Training import train_siamese, train_hard_negative_siamese, compute_embeddings
import torch
import os

def train_siamese_for_dataset(dataset_name, embedding_dim=128, lr=1e-3, epochs=10, num_pairs=5000, batch_size=64, save_dir="../models"):
    loader = DistDataLoader("../data")
    dataset = loader.load_dataset(dataset_name)
    if dataset is None:
        print(f"❌ Failed to load {dataset_name}")
        return

    input_dim = dataset.X.shape[1] if hasattr(dataset.X, "shape") else len(dataset.X[0])
    print(f"📊 Dataset {dataset_name} loaded with {dataset.target_names}")
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim, num_classes=len(dataset.target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"🚀 Training SiameseNetwork for {dataset_name}...")
    train_siamese(
        model,
        dataset,
        optimizer,
        epochs=epochs,
        num_pairs=num_pairs,
        batch_size=batch_size,
        save_dir=save_dir
    )
    print(f"✅ Training complete for {dataset_name}. Model saved in {save_dir}")

def train_hard_negative_siamese_for_dataset(dataset_name, embedding_dim=128, lr=1e-3, epochs=10, num_pairs=5000, batch_size=64, save_dir="../models"):
    loader = DistDataLoader("../data")
    dataset = loader.load_dataset(dataset_name)
    if dataset is None:
        print(f"❌ Failed to load {dataset_name}")
        return

    input_dim = dataset.X.shape[1] if hasattr(dataset.X, "shape") else len(dataset.X[0])
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim, num_classes=len(dataset.target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"🚀 Training soft-negative SiameseNetwork for {dataset_name}...")
    train_siamese(
        model,
        dataset,
        optimizer,
        epochs=epochs,
        num_pairs=num_pairs,
        batch_size=batch_size,
        save_dir=save_dir
    )
    print(f"✅ Soft-negative training complete for {dataset_name}. Model saved in {save_dir}")

    print(f"🔍 Computing embeddings for {dataset_name}...")
    embeddings = compute_embeddings(model, dataset)
    print(f"✅ Embeddings computed for {dataset_name}")
    print(f"🚀 Training hard-negative SiameseNetwork for {dataset_name}...")
    train_hard_negative_siamese(
        model,
        dataset,
        optimizer,
        embeddings=embeddings,
        hard_negative_threshold=1.5,
        epochs=epochs,
        num_pairs=num_pairs,
        batch_size=batch_size,
        save_dir=save_dir
    )
    print(f"✅ Hard-negative training complete for {dataset_name}. Model saved in {save_dir}")

if __name__ == "__main__":
    # Example usage
    train_siamese_for_dataset("iris")