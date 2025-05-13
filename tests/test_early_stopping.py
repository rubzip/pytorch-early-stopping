import sys

import torch.nn as nn

sys.path.append("pytorch_early_stopping")
from early_stopping import EarlyStopping


model = nn.Linear(10, 2)


def test_early_stop_triggered_after_patience():
    """
    Test that early stopping is triggered after the defined patience
    when there is no improvement in the monitored value.
    """
    early_stopping = EarlyStopping(patience=2, verbose=False, higher_is_better=True)
    values = [0.5, 0.5, 0.5, 0.5]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.5


def test_early_stop_not_trigered():
    """
    Test that early stopping is NOT triggered when the monitored value
    improves continuously across epochs.
    """
    early_stopping = EarlyStopping(patience=2, verbose=False, higher_is_better=True)
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 1.0


def test_early_stop_triggered_with_higher_is_better():
    """
    Test early stopping with `higher_is_better=True`:
    Should stop when the monitored value does not improve for 'patience' steps.
    """
    early_stopping = EarlyStopping(patience=2, verbose=False, higher_is_better=True)
    values = [0.5, 0.6, 0.7, 0.6, 0.5]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.7


def test_early_stop_triggered_with_smaller_is_better():
    """
    Test early stopping with `higher_is_better=False`:
    Should stop when the monitored value stops decreasing for 'patience' steps.
    """
    early_stopping = EarlyStopping(patience=2, verbose=False, higher_is_better=False)
    values = [0.5, 0.4, 0.3, 0.4, 0.5]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.3


def test_early_stop_not_triggered_with_smaller_is_better():
    """
    Test that early stopping is NOT triggered when the monitored value
    improves (i.e., decreases) each epoch with `higher_is_better=False`.
    """
    early_stopping = EarlyStopping(patience=2, verbose=False, higher_is_better=False)
    values = [0.5, 0.4, 0.3, 0.2, 0.1]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 0.1


def test_start_from_epoch():
    """
    Test that early stopping does not track improvements until
    reaching `start_from_epoch`. It should ignore early epochs.
    """
    early_stopping = EarlyStopping(
        patience=2, verbose=False, start_from_epoch=5, higher_is_better=True
    )
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9]

    for value in values:
        print(value)
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 0.9


def test_delta():
    """
    Test that `min_delta` is respected: improvements smaller than delta
    are not considered improvements, and patience is still consumed.
    """
    early_stopping = EarlyStopping(
        patience=2, verbose=False, min_delta=1.0, higher_is_better=True
    )
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop is True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.5
