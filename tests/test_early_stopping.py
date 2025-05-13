import torch.nn as nn

import sys
sys.path.append('./pytorch_early_stopping')
from pytorch_early_stopping.early_stopping import EarlyStopping

model = nn.Linear(10, 2)

def test_early_stop_triggered_after_patience():
    early_stopping = EarlyStopping(patience=2, verbose=False)
    values = [0.5, 0.5, 0.5, 0.5]  # No hay mejora

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.5

def test_early_stop_not_trigered():
    early_stopping = EarlyStopping(patience=2, verbose=False, bigger_is_better=True)
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 1.0

def test_early_stop_triggered_with_bigger_is_better():
    early_stopping = EarlyStopping(patience=2, verbose=False, bigger_is_better=True)
    values = [0.5, 0.6, 0.7, 0.6, 0.5]  # Mejora en los primeros dos

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.7

def test_early_stop_triggered_with_smaller_is_better():
    early_stopping = EarlyStopping(patience=2, verbose=False, bigger_is_better=False)
    values = [0.5, 0.4, 0.3, 0.4, 0.5]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.3

def test_early_stop_not_triggered_with_smaller_is_better():
    early_stopping = EarlyStopping(patience=2, verbose=False, bigger_is_better=False)
    values = [0.5, 0.4, 0.3, 0.2, 0.1]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 0.1

def test_start_from_epoch():
    early_stopping = EarlyStopping(patience=2, verbose=False, start_from_epoch=5)
    values = [0., 0., 0., 0., 0., 0., 0.5, 0.6, 0.7, 0.8, 0.9]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == False
    assert early_stopping.epochs_without_improvement == 0
    assert early_stopping.best_value == 0.9

def test_delta():
    early_stopping = EarlyStopping(patience=2, verbose=False, min_delta=1.)
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for value in values:
        early_stopping.step(value=value, model=model)
        if early_stopping.early_stop:
            break

    assert early_stopping.early_stop == True
    assert early_stopping.epochs_without_improvement == 2
    assert early_stopping.best_value == 0.5
