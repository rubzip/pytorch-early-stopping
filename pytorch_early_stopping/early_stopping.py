import copy
from typing import Callable

import torch.nn as nn

class EarlyStopping:
    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 1e-8,
            verbose: bool = True,
            restore_best_weights: bool = False,
            start_from_epoch: int = 0,
            bigger_is_better: bool = True,
            log_func: Callable[[str], None] = print
        ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.bigger_is_better = bigger_is_better
        self.log_func = log_func

        self.epoch = 0
        self.epochs_without_improvement = 0
        self.early_stop = False
        self.best_value = None
        self.best_model = None

    def __call__(self, value: float, model: nn.Module):
        return self.step(value=value, model=model)

    def _log(self, msg: str) -> None:
        if self.verbose:
            self.log_func(msg)

    def _save_best_model(self, model: nn.Module) -> None:
        if self.restore_best_weights:
            self.best_model = copy.deepcopy(model.state_dict())
            self._log(f"Epoch {self.epoch} | Saving best model.")
    
    def restore_best_model(self, model: nn.Module) -> None:
        if self.restore_best_weights and self.best_model is not None:
            model.load_state_dict(self.best_model)
            self._log(f"Epoch {self.epoch} | Restoring best model.")

    def step(self, value: float, model: nn.Module) -> None:
        self.epoch += 1
        if self.early_stop:
            return
        
        if (self.epoch - 1) < self.start_from_epoch:
            return
        
        if self.best_value is None:
            self.best_value = value
            self._save_best_model(model)
            self._log(f"Epoch {self.epoch} | Best score initialized at {value:.6f}. Patience {self.epochs_without_improvement}.")
            return
        
        adjusted_value = value + self.min_delta
        model_improved = adjusted_value > self.best_value if self.bigger_is_better else adjusted_value < self.best_value

        if model_improved:
            self.best_value = value
            self.epochs_without_improvement = 0
            self._save_best_model(model)
            self._log(f"Epoch {self.epoch} | Improved best score to {value:.6f}. Reset patience.")
            return
        
        self.epochs_without_improvement += 1
        self._log(f"Epoch {self.epoch} | No improvement. Patience {self.epochs_without_improvement}/{self.patience}.")

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True
            self._log(f"Epoch {self.epoch} | Patience exhausted. Early stopping...")
            self.restore_best_model(model)
