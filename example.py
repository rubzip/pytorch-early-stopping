from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pytorch_early_stopping.early_stopping import EarlyStopping

class ExampleModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def test_early_stopping_breast_cancer(
        early_stop_on: str = "loss", 
        bigger_is_better: bool = False,
        num_epochs: int = 1_000_000,
        batch_size: int = 64
        ):
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    X_val, y_val = torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

    class_weights = torch.tensor(
        compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.numpy()),
        dtype=torch.float32
        )
    pos_weight = class_weights[1] / class_weights[0]

    model = ExampleModel(X_train.shape[-1])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopper = EarlyStopping(patience=5, restore_best_weights=True, bigger_is_better=bigger_is_better)

    len_train, len_val = len(X_train), len(X_val)
    for epoch in range(num_epochs):
        train_loss, val_loss = 0., 0.
        train_correct, val_correct = 0, 0

        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()

            y_p = model(x).squeeze()
            loss = criterion(y_p, y)

            loss.backward()
            optimizer.step()

            predictions = (y_p >= 0.).float()

            train_loss += loss.item()
            train_correct += (predictions == y).sum().item()

        train_loss /= len_train
        train_acc = train_correct / len_train

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                y_p = model(x).squeeze()
                loss = criterion(y_p, y)
                predictions = (y_p >= 0.).float()

                val_loss += loss.item()
                val_correct += (predictions == y).sum().item()

        val_loss /= len_val
        val_acc = val_correct / len_val

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        value = val_loss if early_stop_on == "loss" else val_acc
        early_stopper(value, model)
        if early_stopper.early_stop:
            break
    return model

if __name__ == "__main__":
    model1 = test_early_stopping_breast_cancer(early_stop_on="loss", bigger_is_better=False)
    model2 = test_early_stopping_breast_cancer(early_stop_on="acc", bigger_is_better=True)
