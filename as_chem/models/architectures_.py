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

        phi_fn: function - see corollary 6 of paper / defaults to an MLP/NN

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
        eps_adj_m = adj_m + I*(1+self.epsilon) # based on paper, shouldnt the epsilon addition be done before the linear? i.e. add the epsilon to the features.
        out = self.phi_fn(torch.matmul(eps_adj_m,h))
        return out


class gat_layer(nn.Module):
    def __init__(self,in_d:int, out_d:int, att_heads:int, dropout:float = 0.2 , att_agg:str = 'concat', nonlinearity:str = 'LeakyRelu'):
        """
        att_K: Number of attention mechanisms - specified as k in the paper.
        att_agg: How to aggregate the attention mechanisms. Typically concatenated unless it is the final layer.
        """
        super(gat_layer, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.linear = nn.Linear(in_d, out_d)
        #attention mechanism
        # Two possible mechanisms: 
        # 1. Create a matrix with all the concatenated combinations as in https://github.com/Diego999/pyGAT/blob/b7b68866739fb5ae6b51ddb041e16f8cef07ba87/layers.py#L42
        # 2. Split the 'a' vector into 2, first will interact with source node, the second will interact with the destination/neighbouring node
        # 2. (continued) add the left and the right
        # Will use the second method for now. Seems more efficient.
        self.att_src = nn.Parameter(torch.tensor(att_heads, out_d)) # might need to add another dimension 
        self.att_dst = nn.Parameter(torch.tensor(att_heads, out_d)) # other implementations had: 1 x H x Out_D
        
        self.leaky_relu =  nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features:torch.tensor, AM:torch.tensor, BI:torch.tensor):
        #TODO: add in dropout steps between all steps

        Wh = self.linear(features)
        att_coef_src_all = torch.matmul(self.att_src *  Wh)
        att_coef_dst_all = torch.matmul(self.att_dst *  Wh)

        # at this point we have the first half(source) 
        # and the second half(destination) 
        # of the attention coefficient for all nodes
        # now we need to add combinations of these together that correspond to edges in the graph

        att_coef_src = att_coef_src_all.index_select(0,BI[:,0])
        att_coef_dst = att_coef_dst_all.index_select(0,BI[:,1])
        
        att_coef = self.leaky_relu(att_coef_src+att_coef_dst)


        # aggregate using attention weights
        att_coef_norm = self.local_normalisation(att_coef, BI, AM.shape[0])

        # Neighbourhood aggregation
        # select the features needed for the attention multiplication
        Wh_bond_selected = Wh.index_select(0,BI[:0]) 
        # selected_features: len_edge_idx x num_heads x out_d
        # selected_normalised coefficients: len_edge_idx x num heads
        # squeeze necessary to match dimensions
        # elemntwise product (Hadamard) gives us len_edge_idx x num_heads x out_d since 1 from coefficients is broadcast
        att_weighted_Wh = Wh_bond_selected * att_coef_norm.unsqueeze(-1)
        updated_features = self.aggregate_neighbourhood(att_weighted_Wh, edge_index, num_nodes)



    def select_edges(self,ac_src, ac_dst, edge_index):
        """

        """

    def local_normalisation(self, ACs, edge_index, num_nodes):
        """
        normalise the coefficients relative to the neighbourhood  
        i.e. coefficients of other nodes connected to the source node
        normalisation done via softmax

        Consider subtracting the max as per 'gordicaleksa/pytorch-GAT' implementation.
        Theory:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """

        # numerator 
        ACs = ACs - ACs.max()
        exp_ACs = ACs.exp()

        # denominator
        neighbourhood_pop_ac = self.ac_neighbourhood_sum(exp_ACs, edge_index[0,], num_nodes)
        
        # increment on denominator to avoid divide by 0
        return(exp_ACs/ (neighbourhood_pop_ac+ 1e-16)


    def ac_neighbourhood_sum(self, exp_ACs, edge_index_src, num_nodes):

        # Expand the edge_index to have a column for every attention head - this is needed for the summation later on
        # (Num_Edges*2) x 1 --> (Num_Edges*2) x AttHeads - each edge is represented twice in the edge index(one for each direction) 
        edge_index_src_expanded  = edge_index_src.unsqueeze(-1).repeat(1, self.att_heads)

        # add all occurences of each att coeff corresponding to each node - repeated for each head
        # should leave us with a Nodes x AttHeads
        node_att_sums = torch.zeros((num_nodes,self.att_heads), dtype=exp_ACs.dtype, device=exp_ACs.device)
        node_att_sums.scatter_add_(0,edge_index_src_expanded,exp_ACs)


        # Since the normalisation is done by dividing over all edge coefficients, we need
        # to expand this to Length_Edge_idx x AttHeads
        # this will have the effect of repeating a value wherever it appears in the index
        edge_att_sums = att_sums.index_select(0,edge_index_src)

        return edge_att_sums


    def aggregate_neighbourhood(weighted_features, edge_index, num_nodes):
        """
        Update features by aggregating the signals from neighbouring nodes.
        """
        # num_nodes * num_att_heads * out_dimensions
        updated_features = torch.zeros((num_nodes,self.att_heads, weighted_features.shape[-1])) 

        #Broadcast across 1. attention heads 2. Features_Out   
        edge_index_src_expanded = edge_index[[0,]].unsqueeze(-1).repeat(1, self.att_heads)
        edge_index_src_expanded = edge_index_src_expanded.unsqueeze(-1).repeat(1, updated_features.shape[-1])


        #sum up weighted neighbours features
        # num_Nodes * num_Heads * num_Out_Features
        updated_features.scatter_add_(0, edge_index_src_expanded,updated_features)

        return updated_features