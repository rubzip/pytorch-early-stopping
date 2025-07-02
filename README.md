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

## API
### Early Stopping
Early stopping class is defined as:
```python
EarlyStopping(
        patience: int = 5,
        min_delta: float = 1e-8,
        verbose: bool = True,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0,
        initial_epoch: int = 0,
        higher_is_better: bool = False,
        log_func: Callable[[str], None] = print,
    )
```
| Argument               | Type                    | Default | Description                                        |
| ---------------------- | ----------------------- | ------- | -------------------------------------------------- |
| `patience`             | `int`                   | `5`     | Number of epochs to wait without improvement       |
| `min_delta`            | `float`                 | `1e-8`  | Minimum change to qualify as improvement           |
| `restore_best_weights` | `bool`                  | `False` | Whether to restore best model after stopping       |
| `start_from_epoch`     | `int`                   | `0`     | Skip early stopping checks for the first N epochs  |
| `higher_is_better`     | `bool`                  | `False` | Whether a higher metric value is considered better |
| `verbose`              | `bool`                  | `True`  | Whether to print logs during training              |
| `log_func`             | `Callable[[str], None]` | `print` | Custom logger (e.g., `logger.info`)                |


## Example of usage
Integrating an EarlyStopping object in your training loop is as easy as:
```python3
from pytorch_early_stopping.early_stopping import EarlyStopping
# Initialize early stopping object
early_stopping = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
    higher_is_better=False
)

for epoch in range(num_epochs):
    # Train model and evaluate it
    ...
    val_loss = compute_validation_loss(...)

    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break
```

## Tests
Run the test suite using pytest:

```bash
pip install pytest
pytest tests/test_early_stopping.py
```
## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/rubzip/pytorch-early-stopping/blob/main/LICENSE) file for details.
