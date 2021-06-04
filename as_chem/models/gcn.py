from typing import List
from functools import partial

import torch
import torch.nn as nn

from .gnn_base import GNN
from .utils import supported_activations

class gcn_layer(nn.Module):
    def __init__(self, in_d:int, out_d:int):
        super(gcn_layer, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.linear = nn.Linear(in_d,out_d)

    def forward(self, features, bonds):
        h = self.linear(features)
        h = torch.matmul(bonds,h)

        return h

"""class masked_linear(nn.Module):
    def __init__(self, in_d:int, out_d:int):
        super(masked_linear, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.weight = nn.Linear(in_d,out_d)
        
        def forward(self, h, mask):
        h = torch.matmul(mask,h)
        h = self.weight(h)
        return" h
"""

class GCN(nn.Module):
    def __init__(self, input_features:int, hidden_channels:List[int], activation:str, dropout_perc:float, out_classes:int):
        super(GCN, self).__init__()
        self.convolutions = nn.ModuleList()
        in_d, out_d = input_features, None
        for d in hidden_channels:
            out_d = d 
            conv_layer = gcn_layer(in_d, out_d)
            self.convolutions.append(conv_layer)
            in_d = out_d
        

        assert activation in supported_activations, f'Activation not supported. Choose from {[a for a in activations.keys()]}'
        self.nonlinearity = supported_activations[activation]
        self.nonlinearity_name = activation
        self.dropout = nn.Dropout(dropout_perc)
        self.fc = gcn_layer(out_d, out_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, A, graph_sizes):

        for conv in self.convolutions:
            h = conv(X, A)
            h = self.nonlinearity(h)
            X = self.dropout(h)
        
        out =  self.fc(X,A)
        out = self.sigmoid(out)
        out = torch.stack([g.mean() for g in out.split(graph_sizes)])
        out.retain_grad()
        return (out)
    
    def init_weights(self):
        for n,p in self.named_parameters():
            if n.endswith('weight'):
                if self.nonlinearity_name in ['relu', 'leaky_relu']:
#                     nn.init.kaiming_uniform_(p.data,nonlinearity=self.nonlinearity_name)
                    nn.init.xavier_uniform_(p.data)

                else:
                    nn.init.xavier_uniform_(p.data)
                    

class GCN_generic(GNN):
    def __init__(self, n_input_features: int, hidden_channels: List[int], 
                activation:str, dropout_perc: float,out_classes: int):
                
                super(GCN_generic, self).__init__(
                    architecture=gcn_layer, 
                    input_features = n_input_features, 
                    hidden_channels = hidden_channels,
                    att_heads=1, 
                    activation =activation,
                    dropout =dropout_perc, 
                    out_classes =out_classes )
                                        

