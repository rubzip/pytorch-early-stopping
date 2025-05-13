import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('./pytorch_early_stopping')
from early_stopping import EarlyStopping

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flaten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flaten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train(
        early_stop_on: str = "loss", 
        bigger_is_better: bool = False,
        num_epochs: int = 1_000_000,
    ) -> nn.Module:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping
    early_stopper = EarlyStopping(patience=5, restore_best_weights=True, bigger_is_better=bigger_is_better)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch+1} | Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        if early_stop_on == "loss":
            score = val_loss
        if early_stop_on == "acc":
            score = val_acc
        early_stopper(score, model)

        if early_stopper.early_stop:
            break

if __name__ == "__main__":
    train(early_stop_on="loss", bigger_is_better=False)
    train(early_stop_on="acc", bigger_is_better=True)
