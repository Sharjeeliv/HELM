# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
from time import time
from typing import Literal

# External
import torch
from torch import nn
from optuna import Trial
import optuna

# Relative
from .utils.utils import Stage, EarlyStopping, epoch_print, run_hook
from .test import evaluate


# #############################
# FUNCTIONS: HELPER
# #############################
def _stopper(es: EarlyStopping, trial: Trial|None) -> bool:
    if trial and trial.should_prune():  raise optuna.TrialPruned()
    if not (es and es.early_stop):      return False 
    print(f"Early stop!", sep=' ')
    return True

def _trainer(dataset, model, optimizer, criterion):
    X, y, m = dataset['X'], dataset['y'], dataset['tr_mask']
    if m is None: raise Exception("No mask provided with key 'tr_mask'!")
    
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
def loop(model: nn.Module, criterion, optimizer, dataset, 
         epochs: int, stage: Stage, trial: Trial|None =None
         ) -> dict[str, object]:
    
    # Move everything to the same device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    criterion.to(DEVICE)
    dataset['X'] = dataset['X'].to(DEVICE)
    dataset['y'] = dataset['y'].to(DEVICE)
    dataset['tr_mask'] = dataset['tr_mask'].to(DEVICE)
    dataset['te_mask'] = dataset['te_mask'].to(DEVICE)
    dataset['va_mask'] = dataset['va_mask'].to(DEVICE)
    
    total_time = 0.0
    early_stopping = EarlyStopping()
    training_history = []
    for epoch in range(1, epochs + 1):
        
        start_time = time()
        # Hook: Run pre-forward logic if provided
        run_hook(dataset, 'prop', model)
        
        # Run training step and evaluation
        tr_loss = _trainer(dataset, model, optimizer, criterion)
        results = evaluate(dataset, model, criterion, stage)
        
        # Timing & Printing
        epoch_time = time() - start_time
        total_time += epoch_time
        
        if epoch % 10 == 0: 
            l, a = results['l'], results['a']
            epoch_print(epoch, tr_loss, l, a, epoch_time)
        
        if not trial and epoch % 10 == 0: 
            tmp = {'epoch': epoch, 'loss': tr_loss, 'train_acc': results['a'].item()}
            training_history.append(tmp)
        
        # Early Stopping
        if _stopper(early_stopping, trial): break
    
    print(f"Total Training Time: {total_time:.2f}s")
    if stage == Stage.TUNE: return {'l': results['l'], 'a': results['a']}
    return {"metrics": results, "history": training_history, "time": total_time}
