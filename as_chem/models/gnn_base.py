from typing import List
from functools import partial

import torch
import torch.nn as nn

class GNN(nn.Module):
    """
        base class for all graph convolution/ message passing networks.

        #TODO finalise initialisation parameters:
           - out_classes? maybe task type instead
           - hidden channels applicable to all networks?
           - ...

        #TODO build a class that works with 4 types of networks (GCN original, GIN, GraphSage, GAT)
        
        #TODO identify generic components and specific components (
           at this point forward pass is only specific identified- double check)


    """
    def __init__(self, architecture:nn.Module ,input_features:int, hidden_channels:List[int], activation:str,
                 dropout:float, out_classes:int ):
                
        super(GNN, self).__init__()
        self.gnn_layer = architecture
        in_d, out_d = input_features, None
        self.graph_convs = nn.ModuleList()
        for d in hidden_channels:
                    out_d = d 
                    gnn_layer = self.gnn_layer(in_d, out_d)
                    self.graph_convs.append(gnn_layer)
                    in_d = out_d                
        
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid
        }

        assert activation in activations, f'Activation not supported. Choose from {[a for a in activations.keys()]}'
        self.nonlinearity = activations[activation]()
        self.nonlinearity_name = activation
        self.dropout = nn.Dropout(dropout)
        self.fc = self.gnn_layer(out_d, out_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, A, graph_sizes):
        for g_conv in self.graph_convs:
            h = g_conv(X, A)
            h = self.nonlinearity(h)
            X = self.dropout(h)
        
        out =  self.fc(X,A)
        out = self.sigmoid(out)
        out = torch.stack([g.mean() for g in out.split(graph_sizes)])
        out.retain_grad()
        return out

    def init_weights(self):
        for n,p in self.named_parameters():
            if n.endswith('weight'):
                if self.nonlinearity_name in ['relu', 'leaky_relu']:
#                     nn.init.kaiming_uniform_(p.data,nonlinearity=self.nonlinearity_name)
                    nn.init.xavier_uniform_(p.data)

                else:
                    nn.init.xavier_uniform_(p.data)
