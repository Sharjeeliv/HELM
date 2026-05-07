# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin

# External
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support as class_metrics)

# Relative
from .utils.utils import Stage

# #############################
# FUNCTIONS: HELPER
# #############################
def _metrics(preds, labels, loss):
    a = accuracy_score(labels, preds)
    p, r, f1, _ = class_metrics(labels, preds, average='macro', zero_division=0)
    metrics = {'a': a, 'p': p, 'r': r, 'f': f1, 'l': loss, 
               'preds': preds, 'labels': labels}
    return metrics

def _get_mask(dataset, stage: Stage):
    # We use va_mask for tuning, for training we use
    # te_mask to see progress, final eval is te_mask
    if stage == Stage.TUNE:  return dataset['va_mask']
    if stage == Stage.TRAIN: return dataset['te_mask']
    if stage == Stage.TEST:  return dataset['te_mask']
    raise ValueError(f"Invalid stage: {stage}")


# #############################
# FUNCTIONS: INTERFACE
# #############################
def evaluate(dataset, model, criterion, stage: Stage):
    X, y = dataset['X'], dataset['y']
    m = _get_mask(dataset, stage)
    
    model.eval()
    with torch.no_grad():
        preds = model(X)
        l = criterion(preds[m], y[m])
        a = (preds[m].argmax(1) == y[m]).float().mean()

    # Return select metrics based on stage
    if stage == Stage.TRAIN:    return {'l': l, 'a': a}
    if stage == Stage.TEST:     return {'l': l, 'a': a}

    # Return full metrics for testing stage
    preds = preds[m].argmax(1).cpu().numpy()
    labels = y[m].cpu().numpy()
    return _metrics(preds, labels, l)


def test(model, dataset):
    criterion = torch.nn.CrossEntropyLoss()
    return evaluate(dataset, model, criterion, stage=Stage.TEST)