# pytorch-early-stopping
A lightweight and flexible early stopping utility for PyTorch training loops. Supports custom patience, delta, and automatic model weight restoration.

Easily plug it into your existing training workflow to prevent overfitting.

## Installation
Through pip:
```bash
pip install pytorch-early-stopping
```
Or install directly from source:
```bash
pip install git+https://github.com/rubzip/pytorch-early-stopping.git
```

## Example of usage
```python3
from pytorch_early_stopping import EarlyStopping

early_stopping = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
    higher_is_better=False
)

for epoch in range(num_epochs):
    # Train model and evaluate it
    ...

    val_loss = compute_validation_loss()
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break
```
