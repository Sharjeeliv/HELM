# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
import json
from time import time

# External
import torch
import optuna

# Relative
from .utils.utils import EarlyStopping, epoch_print, Stage
from .test import evaluate


# #############################
# FUNCTIONS: HELPER
# #############################
def _stopper(es: EarlyStopping, kwargs: dict):
    trial = kwargs.get('trial', None)
    if trial and trial.should_prune():  raise optuna.TrialPruned()
    if not (es and es.early_stop):      return False 
    print(f"Early stop!", sep=' ')
    return True

def _trainer(dataset, model, optimizer, criterion):
    X, y, m = dataset['X'], dataset['y'], dataset['mask']
    if m is None: raise Exception("No mask provided with key 'mask'!")
    
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out[m], y[m])
    loss.backward()
    optimizer.step()
    return loss.item()


# #############################
# FUNCTIONS: INTERFACE
# #############################
def loop(model, criterion, optimizer, dataset, epochs, stage, kwargs: dict):
    
    total_time = 0.0
    early_stopping = EarlyStopping()
    is_tune = kwargs.get('trial', None) is not None
    training_history = []
    for epoch in range(1, epochs + 1):
        
        start_time = time()
        tr_loss = _trainer(dataset, model, optimizer, criterion)
        res = evaluate(dataset, model, criterion, kwargs=kwargs)
        
        # Timing & Printing
        epoch_time = time() - start_time
        total_time += epoch_time
        if epoch % 10 == 0: epoch_print(epoch, tr_loss, res['l'], res['a'], epoch_time)
        
        if not is_tune and epoch % 10 == 0: 
            tmp = evaluate(dataset, model, criterion, kwargs={'full': True, **kwargs})
            training_history.append({'epoch': epoch, 'loss': tr_loss, 'train_acc': res['a']})
        
        # Early Stopping
        if _stopper(early_stopping, kwargs): break
    
    print(f"Total Training Time: {total_time:.2f}s")
    return res if is_tune else {**res, 'training_time': total_time}
