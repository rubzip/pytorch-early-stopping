import copy
import math
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
        initial_epoch: int = 0,
        higher_is_better: bool = False,
        log_func: Callable[[str], None] = print,
    ):
        """A flexible and lightweight early stopping utility for PyTorch training loops.

        This class monitors a validation metric (e.g., loss or accuracy) during training and stops the loop
        if no significant improvement is observed after a specified number of epochs (patience). Also,
        it can restore the model weights corresponding to the best metric value seen.

        Args:
            patience (int, optional): Number of consecutive epochs without improvement to wait before training stopping. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored value to qualify as an improvement. Defaults to 1e-8.
            verbose (bool, optional): If True, prints logging messages during training. Defaults to True.
            restore_best_weights (bool, optional): If True, restores model weights from the epoch with the best value upon stopping. Defaults to False.
            start_from_epoch (int, optional): Number of initial epochs to skip before starting early stopping checks. Defaults to 0.
            initial_epoch (int, optional): Value to initialize epoch count. Defaults to 0.
            higher_is_better (bool, optional): If True, a higher value is considered an improvement (e.g., accuracy).
                                 If False, a lower value is considered better (e.g., loss). Defaults to False.
            log_func (Callable[[str], None], optional): Logging function to use (e.g., `print`, `logger.info`). Defaults to print.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.higher_is_better = higher_is_better
        self.log_func = log_func
        self.epoch = initial_epoch

        self.epochs_without_improvement = 0
        self.early_stop = False
        self.best_value = None
        self.best_model = None

    def __call__(self, value: float, model: nn.Module):
        """Alias for the step method to allow the object to be called directly.

        Args:
            value (float): The current validation metric (e.g., loss or accuracy).
            model (nn.Module): The PyTorch model being trained.
        """
        return self.step(value=value, model=model)

    def _log(self, msg: str) -> None:
        """Logs a message using the provided logging function, if verbose is enabled.

        Args:
            msg (str): Message to be logged.
        """
        if self.verbose:
            self.log_func(msg)

    def _save_best_model(self, model: nn.Module) -> None:
        """Saves the current model's state_dict as the best model if restore_best_weights is True.

        Args:
            model (nn.Module): The PyTorch model being trained.
        """
        if self.restore_best_weights:
            self.best_model = copy.deepcopy(model.state_dict())
            self._log(f"Epoch {self.epoch} | Saving the current model.")

    def _improvement_achieved(self, value, model) -> None:
        self.best_value = value
        self.epochs_without_improvement = 0
        self._save_best_model(model)

    def restore_best_model(self, model: nn.Module) -> None:
        """Restores the model weights to the best recorded state.
        This is only done if `restore_best_weights` is True and a best model has been saved.

        Args:
            model (nn.Module): The PyTorch model to restore.
        """
        if self.restore_best_weights and self.best_model is not None:
            model.load_state_dict(self.best_model)
            self._log(f"Epoch {self.epoch} | Restoring best model.")

    def step(self, value: float, model: nn.Module) -> None:
        """Performs one step of early stopping evaluation.
        Monitors the validation metric to determine if training should be stopped based on
        lack of improvement. Optionally saves and restores the best model state.

        Args:
            value (float): Current value of the monitored validation metric.
            model (nn.Module): The PyTorch model being trained.
        """
        self.epoch += 1
        if self.early_stop:
            return

        if math.isnan(value):
            return

        if (self.epoch - 1) < self.start_from_epoch:
            return

        if self.best_value is None:
            self._improvement_achieved(value, model)
            self._log(
                f"Epoch {self.epoch} | Best score initialized at {value:.6f}. Patience {self.epochs_without_improvement}."
            )
            return

        adjusted_value = value - self.min_delta
        model_improved = (
            adjusted_value >= self.best_value
            if self.higher_is_better
            else adjusted_value <= self.best_value
        )

        if model_improved:
            self._improvement_achieved(value, model)
            self._log(
                f"Epoch {self.epoch} | Improved best score to {value:.6f}. Reset patience."
            )
            return

        self.epochs_without_improvement += 1
        self._log(
            f"Epoch {self.epoch} | No improvement. "
            f"Patience {self.epochs_without_improvement}/{self.patience}."
        )

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True
            self._log(f"Epoch {self.epoch} | Patience exhausted. Early stopping...")
            self.restore_best_model(model)
