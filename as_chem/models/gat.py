from typing import List
from functools import partial

import torch
import torch.nn as nn

from .gnn_base import GNN
from .utils import supported_activations

class GAT_generic(GNN):
    def __init__(self, *args, **kwargs):
                
                super(GCN_generic, self).__init__(*args, **kwargs)
                                        
