from typing import List

import torch 
import torch.nn as nn

from .utils import supported_activations
from .architectures import gin_layer
from .gnn_base import GNN

class GIN(nn.Module):
    def __init__(self, n_input_features: int, hidden_channels: List[int], 
                activation:str, dropout_perc: float,out_classes: int):
                super(GIN, self).__init__()
                self.graph_convs = nn.ModuleList()
                in_d, out_d = n_input_features, None
                for d in hidden_channels:
                    out_d = d 
                    graph_conv_layer = gin_layer(in_d, out_d)
                    self.graph_convs.append(graph_conv_layer)
                    in_d = out_d
                
                assert activation in supported_activations, f'Activation not supported. Choose from {[a for a in activations.keys()]}'
                self.nonlinearity = supported_activations[activation]
                self.dropout = nn.Dropout(dropout)
                self.fc = gin_layer(out_d, out_classes)
                self.sigmoid = nn.Sigmoid()

    def forward(self, X, A, graph_sizes): # figure out a way to abstract the graph sizes
        # why cant all this be abstracted into base class?
        # the only diff is the GNN layer
        for conv in self.graph_convs:
            h = conv(X, A)
            h = self.nonlinearity(h)
            X = self.dropout(h)

        out = self.fc(X, A)
        out = self.sigmoid(out)
        out = torch.stack([g.mean() for g in out.split(graph_sizes)])
        out.retain_grad()
        return (out)

    def init_weights(self): # all this can be moved into base class?
        for n,p in self.named_parameters():
            if n.endswith('weight'):
                if self.nonlinearity_name in ['relu', 'leaky_relu']:
#                     nn.init.kaiming_uniform_(p.data,nonlinearity=self.nonlinearity_name)
                    nn.init.xavier_uniform_(p.data)

                else:
                    nn.init.xavier_uniform_(p.data)
                    

class GIN_generic(GNN):
    def __init__(self, n_input_features: int, hidden_channels: List[int], 
                activation:str, dropout_perc: float,out_classes: int):
                
                super(GIN_generic, self).__init__(
                    architecture=gin_layer, 
                    input_features = n_input_features, 
                    hidden_channels = hidden_channels, 
                    activation =activation,
                    dropout =dropout_perc, 
                    out_classes =out_classes )
                                        

