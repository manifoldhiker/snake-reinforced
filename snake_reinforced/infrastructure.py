import os
import random
import numpy as np
import torch
from pathlib import Path


def seed_all(seed):
    print('Setting global seed to', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_parent_exists(filename):
    Path(filename).parent.mkdir(exist_ok=True, parents=True)


def save_state_dict(net, filename):
    ensure_parent_exists(filename)
    torch.save(net.state_dict(), filename)
