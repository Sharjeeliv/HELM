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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as class_metrics

# Relative
from .utils.optuna import get_model, get_optimizer


# #############################
# FUNCTIONS: HELPER
# #############################
def _metrics(preds, labels, loss):
    a = accuracy_score(labels, preds)
    p, r, f1, _ = class_metrics(labels, preds, average='macro', zero_division=0)
    metrics = {'a': a, 'p': p, 'r': r, 'f': f1, 'l': loss, 
               'preds': preds, 'labels': labels}
    return metrics


# #############################
# FUNCTIONS: INTERFACE
# #############################
def evaluate(dataset, model, criterion, kwargs: dict):
    X, y, m = dataset['X'], dataset['y'], dataset['mask']
    if m is None: raise Exception("No mask provided with key 'mask'!")
    
    model.eval()
    with torch.no_grad():
        preds = model(X)
        l = criterion(preds[m], y[m])
        a = (preds[m].argmax(1) == y[m]).float().mean()

    if kwargs.get('trial', False):     return {'l': l, 'a': a}
    if not kwargs.get('full', False):  return {'l': l, 'a': a}
    
    preds = preds[m].argmax(1).cpu().numpy()
    labels = y[m].cpu().numpy()
    return _metrics(preds, labels, l)


def test(model, dataset, kwargs: dict):
    criterion = torch.nn.CrossEntropyLoss()
    return evaluate(dataset, model, criterion, kwargs={'full': True, **kwargs})