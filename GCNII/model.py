import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features
        
        self.out_features = out_features

        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 /math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj, h_0, lamda, alpha, layer_index):
        beta_l = math.log(lamda/ layer_index + 1)
        h_i = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1-alpha) * h_i + alpha*h0
        else:
            support = (1-alpha) * h_i + alpha*h_0
            r = support
        output = beta_l * torch.mm(support, self.weight) + (1-beta_l) * r 
        if self.residual:
            output = output + input
        return output

class GCNII(nn.Module):
    def __init__(self, num_feature, num_layers, num_hidden, num_class, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.list_convs = nn.ModuleList()
        for i in range(num_layers):
            self.list_convs.append(GraphConvolution(num_hidden, num_hidden, variant=variant))
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(num_feature, num_hidden))
        self.fc_layers.append(nn.Linear(num_hidden, num_class))

        self.params1 = list(self.list_convs.parameters())
        self.params2 = list(self.fc_layers.parameters())

        self.activation = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
    
    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.activation(self.fc_layers[0](x))
        _layers.append(layer_inner)
        for i, conv in enumerate(self.list_convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.activation(conv(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fc_layers[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
