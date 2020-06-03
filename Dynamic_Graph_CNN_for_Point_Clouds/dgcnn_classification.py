import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data/ModelNet10')

def MLP(channels, batch_norm=True):
    net = layers[]
    for i in range(1, len(channels):
        net.append(Linear(channels[i-1], channels[i]))
        net.append(ReLU())
        net.append(BN(channels[i]))
    return Seq(*net)


