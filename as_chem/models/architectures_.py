from typing import Callable, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



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

class gin_layer(nn.Module):
    def __init__(self, in_d:int, out_d:int, phi_fn:Union[None,Callable]= None):

        """

        phi_fn: function - see corollary 6 of paper

        """
        super(gin_layer, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.linear = nn.Linear(in_d, out_d)
        eps_start = nn.init.uniform_(torch.empty(1),-1.,1.)
        self.epsilon = Parameter(eps_start).cuda()
        if phi_fn:
            self.phi_fn = phi_fn
        else:
            self.phi_fn = nn.Sequential( # check the dimensions and logic
                    nn.Linear(out_d, out_d),
                    nn.ReLU(),
                    nn.Linear(out_d,out_d)
            )
    def forward(self, features, adj_m):
        h = self.linear(features)
        I = torch.eye(adj_m.size()[0], device='cuda')
        eps_adj_m = adj_m + I*(1+self.epsilon)
        out = self.phi_fn(torch.matmul(eps_adj_m,h))
        return out
        