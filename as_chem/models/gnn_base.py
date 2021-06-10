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
                 dropout:float, num_tasks:int, **kwargs ):
                
        super(GNN, self).__init__()
        self.gnn_layer = architecture
        in_d, out_d = input_features, None
        self.graph_convs = nn.ModuleList()

        if 'att_heads' in kwargs: #Attention Models
            K = kwargs.get('att_heads')
            del kwargs['att_heads'] # this will be passed in manually
            if type(K) == int:
                K = [K] * len(hidden_channels)
            assert len(K)==len(hidden_channels), "The number of attention heads should be either 1 or one more than the number of hidden channels(for the final layer)."     
            K.insert(0,1) # since raw data has no heads
            for k,d in enumerate(hidden_channels):
                out_d = d
                att_agg = k < (len(hidden_channels)-1) #concat attention heads except final hidden channel
                gnn_layer = self.gnn_layer(in_d*K[k], out_d,att_heads=K[k+1] , att_concat= att_agg, **kwargs)
                self.graph_convs.append(gnn_layer)
                in_d = out_d                
            self.final = nn.Linear(out_d, num_tasks) # no need to use graph structure in final layer

        else:    
            for d in hidden_channels:
                        out_d = d 
                        gnn_layer = self.gnn_layer(in_d, out_d, *args, **kwargs)
                        self.graph_convs.append(gnn_layer)
                        in_d = out_d                
            self.final = nn.Linear(out_d, num_tasks)  # no need to use graph structure in final layer
            
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid
        }

        assert activation in activations, f'Activation not supported. Choose from {[a for a in activations.keys()]}'
        self.nonlinearity = activations[activation]()
        self.nonlinearity_name = activation
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def forward(self, X, A, graph_sizes):
        for g_conv in self.graph_convs:
            h = g_conv(X, A)
            h = self.nonlinearity(h)
            X = self.dropout(h)
        
        out =  self.final(X)
        out = self.sigmoid(out)
        out = torch.stack([g.mean(0) for g in out.split(graph_sizes)]) # split into batches and then aggregate across nodes via averaging
        out.retain_grad()
        return out

    
    def init_weights(self):
        for n,p in self.named_parameters():
            print(n)
            if n.endswith('weight') or ('att_' in n): # attention mechanism isnt done through nn.Linear
                if self.nonlinearity_name in ['relu', 'leaky_relu']:
                     nn.init.kaiming_uniform_(p.data,nonlinearity=self.nonlinearity_name)
#                    nn.init.xavier_uniform_(p.data)

                else:
                    nn.init.xavier_uniform_(p.data)

            if n.endswith('bias'):
                torch.nn.init.zeros_(p.data)