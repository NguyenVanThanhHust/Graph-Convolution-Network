import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
	"""
	Simple GCN layer, as in tutorial
	"""
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(self.out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.weigth.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj_matrix):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj_matrix, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'			

