import sys
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append("pytorch_early_stopping")
from early_stopping import EarlyStopping


class MNISTModel(nn.Module):
    """Simple feedforward neural network for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def run(
    early_stop_on: Literal["loss", "acc"] = "loss", higher_is_better: bool = False
) -> nn.Module:
    """
    Trains a simple MNIST classifier with early stopping.

    Args:
        early_stop_on (str): Metric to monitor ("loss" or "acc").
        higher_is_better (bool): True for acc (higher is better). False for loss (lower is better).
        num_epochs (int): Maximum number of training epochs.

    Returns:
        nn.Module: The trained model.
    """
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopper = EarlyStopping(
        patience=5, restore_best_weights=True, higher_is_better=higher_is_better
    )

    num_epochs = 1_000_000_000
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = train_correct / total

        # Validation
        model.eval()
        val_loss, val_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = val_correct / total

        print(
            f"Epoch {epoch+1} | "
            f"Train: Loss = {train_loss:.4f}, Acc = {train_acc:.4f} | "
            f"Val: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}"
        )

        metrics = {"loss": val_loss, "acc": val_acc}
        early_stopper(metrics[early_stop_on], model)

        if early_stopper.early_stop:
            break

    return model


if __name__ == "__main__":
    run(early_stop_on="loss", higher_is_better=False)
    run(early_stop_on="acc", higher_is_better=True)
