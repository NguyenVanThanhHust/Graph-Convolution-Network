import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn import global_max_pool

from knn_graph import knn_graph
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data/ModelNet10')

def MLP(channels, batch_norm=True):
    net = layers[]
    for i in range(1, len(channels):
        net.append(Linear(channels[i-1], channels[i]))
        net.append(ReLU())
        net.append(BN(channels[i]))
    return Seq(*net)

class EdgeConv(MessagePassing):
    """
    Egde Convolution operator from "Dynamic Graph CNN for
    Learning on Point Clouds"
    Args:
        nn(torch.nn.Module): A neural network that maps pair-wise concatneted feature 
        of shape [-1, 2*in_channels] to shape [-1, out_channels]
        aggr(string): aggregation sheme : add, mean, max
        **kwargs: additional argument
    """
    def __init__(self, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def forward(self, x, edge_index):
        """
        Args: 
            x: tensor: 
            edge_index: 
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j - x_i], dim=1))
    
    def __repr__(self):
        """
        get official represetation of class
        Eg: EdgeConv(nn=)
        """
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class DynamicEdgeConv(EdgeConv):
    """

    """
    def __init__(self, nn, k, aggr='max', **kwargs):
        super(DynamicEdgeConv, self).__init__(nn=nn, aggr=aggr, **kwargs)
        if knn_graph is None:
            raise ImportError("Import knn_graph from torch_cluster or knn_graph.py")
        self.k = k
    
    def forward(self, x, batch=None):
        edge_index= knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        
