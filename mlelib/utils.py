import torch
import os
import numpy as np

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_val_split_indices(n, val_perc=0.25, random_state=None):
    assert val_perc < 1
    assert val_perc > 0

    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_train = int((1-val_perc)*n)
    return indices[:n_train], indices[n_train:]