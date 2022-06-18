import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
				+ str(self.in_features) + ' -> ' \
				+ str(self.out_features) + ')'

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, adj)
		x = F.softmax(x, dim=1)
		return torch.log(x)

class GCN_wEmb(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN_wEmb, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		emb = torch.spmm(adj, x)
		x = self.gc2(x, adj)
		x = F.softmax(x, dim=1)
		return torch.log(x), emb

class SGC(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(SGC, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = self.gc1(x, adj)
		x = self.gc2(x, adj)
		return F.log_softmax(x, dim=1)

class APPNP(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(APPNP, self).__init__()

		self.linear1 = nn.Linear(nfeat, nhid)
		self.linear2 = nn.Linear(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		alpha = 0.2
		x = F.relu(self.linear1(x))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.linear2(x)

		x_k = x

		for _ in range(10):
			x_k = alpha*x + (1-alpha)*torch.mm(adj, x_k)

		return F.log_softmax(x_k, dim=1)
