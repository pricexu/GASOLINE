import os
import time
import json
import random
import argparse
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.autograd import Variable
from copy import deepcopy
from utils import *
from models import GCN, SGC, APPNP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Settings
parser = argparse.ArgumentParser()

# the setting of hyper-gradient computing can be set as default
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--training_epochs', type=int, default=300,
					help='Number of epochs to meta train.')
parser.add_argument('--lr', type=float, default=0.01,
					help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
					help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
					help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
					help='Dropout rate (1 - keep probability).')

# the setting of sanitiion can be customized
parser.add_argument('--modify_topology', default=True,
					help='Whether sanitize topology.')
parser.add_argument('--modify_feature', default=True,
					help='Whether sanitize node feature.')
parser.add_argument('--sanitized_ratio_topology', type=float, default=0.1,
					help='Percentage of edges to be sanitized.')
parser.add_argument('--sanitized_ratio_feature', type=float, default=0.001,
					help='Percentage of features to be sanitized.')
parser.add_argument('--topology_mode', type=str, default='flip', choices=['flip', 'delete', 'add', 'set'],
					help="Mode of topology sanitizations. Select from 'flip', 'delete', 'add', 'set' (i.e., continuous).")
parser.add_argument('--feature_mode', type=str, default='set', choices=['flip', 'delete', 'add', 'set'],
					help="Mode of node feature sanitizations. Select from 'flip', 'delete', 'add', 'set' (i.e., continuous).")
parser.add_argument('--updating_steps', type=int, default=10,
					help='Number of steps to run out of the budget.')
parser.add_argument('--num_fold', type=int, default=4,
					help='Number of folds to generate hyper-gradient.')
parser.add_argument('--backbone_model', type=str, default='GCN', choices=['GCN', 'SGC', 'APPNP'],
					help="The backbone model used to sanitize the graph. Select from 'GCN', 'SGC', 'APPNP'.")
parser.add_argument('--input_data', type=str, default='../data/citeseer',
					help="The name of input file")
parser.add_argument('--output_data', type=str, default='../data/citeseer',
					help="The name of output file")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

def train(model, idx_train, adj_tensor, featuer_tensor, labels, optimizer):
	model.train()
	output = model(featuer_tensor, adj_tensor)
	loss_train = F.nll_loss(output[idx_train], torch.LongTensor(labels[idx_train]))
	optimizer.zero_grad()
	loss_train.backward()
	optimizer.step()

def test(model, idx_test, adj_tensor, featuer_tensor, labels):
	model.eval()
	output = model(featuer_tensor, adj_tensor)
	loss_test = F.nll_loss(output[idx_test], torch.LongTensor(labels[idx_test]))
	acc_test = accuracy(output[idx_test], torch.LongTensor(labels[idx_test]))
	return loss_test, acc_test

def meta_gradient(adj, features, labels, idx_train, idx_val, modify_topology, modify_feature, backbone):

	if isinstance(adj, np.matrix):
		adj_tensor = torch.FloatTensor(adj)
	else:
		adj_tensor = sparse_mx_to_torch_tensor(adj)
	if isinstance(features, np.matrix):
		features_tensor = torch.FloatTensor(features)
	else:
		features_tensor = sparse_mx_to_torch_tensor(features)
	adj_normalized_tensor = preprocess_adj_tensor(adj_tensor)

	adj_difference = torch.zeros_like(adj_tensor)
	feature_difference = torch.zeros_like(features_tensor)

	if backbone == 'GCN':
		backbone = GCN(nfeat=features.shape[1],
					nhid=args.hidden,
					nclass=nclass,
					dropout=args.dropout)
	elif backbone == 'SGC':
		backbone = SGC(nfeat=features.shape[1],
					nhid=args.hidden,
					nclass=nclass,
					dropout=args.dropout)
	elif backbone == 'APPNP':
		backbone = APPNP(nfeat=features.shape[1],
					nhid=args.hidden,
					nclass=nclass,
					dropout=args.dropout)

	optimizer_backbone = optim.Adam(backbone.parameters(),
						   lr=args.lr, weight_decay=args.weight_decay)

	adj_improved = adj_tensor.detach()
	adj_improved.requires_grad_(True)
	adj_normalized_improved = preprocess_adj_tensor(adj_improved)

	feature_improved = features_tensor.detach()
	feature_improved.requires_grad_(True)

	######################### pre-train to get truncate meta-gradients #########################
	for epoch in range(200):
		train(backbone, idx_train, adj_normalized_tensor, features_tensor, labels, optimizer_backbone)

	######################### Obtain truncated meta-gradients #########################

	# change the number to obtain high-order meta-gradients
	for epoch in range(1):

		backbone.train()
		output = backbone(features_tensor, adj_normalized_tensor)
		loss_train = F.nll_loss(output[idx_train], torch.LongTensor(labels[idx_train]))
		optimizer_backbone.zero_grad()
		loss_train.backward(retain_graph=False)
		optimizer_backbone.step()

		output = backbone(feature_improved, adj_normalized_improved)
		loss_self = F.nll_loss(output[idx_val], torch.LongTensor(labels[idx_val]))
		loss_self.backward(retain_graph=True)

		if modify_topology:
			adj_difference -= adj_improved.grad.data+adj_improved.grad.data.transpose(0,1)-torch.diag(torch.diagonal(adj_improved.grad.data, 0))
			adj_improved.grad.zero_()

		if modify_feature:
			feature_difference -= feature_improved.grad.data
			feature_improved.grad.zero_()
	if modify_topology:
		adj_difference = adj_difference.data
	else:
		adj_difference = torch.zeros_like(adj_tensor)

	if modify_feature:
		feature_difference = feature_difference.data
	else:
		feature_difference = torch.zeros_like(features_tensor)

	return adj_difference, feature_difference

if __name__ == "__main__":

	backbone_model = args.backbone_model # 'GCN', SGC', 'APPNP'
	modify_topology = args.modify_topology
	modify_feature = args.modify_feature
	adj_mode = args.topology_mode # 'flip', 'delete', 'add', 'set'
	feature_mode = args.feature_mode # 'flip', 'delete', 'add', 'set'
	sanitized_ratio_topology = args.sanitized_ratio_topology
	sanitized_ratio_feature = args.sanitized_ratio_feature
	steps = args.updating_steps
	num_fold = args.num_fold

	# data_folder = args.input_folder
	input_data = args.input_data
	output_data = args.output_data

	if not modify_topology and not modify_feature:
		print('Should at least modify either topology or feature.')
		sys.exit()

	print('Sanitized ratio of topology: '+str(sanitized_ratio_topology))

	adj, features, labels = load_data(input_data +'.npz')
	nclass = max(labels) + 1

	idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1,
							test_size=0.8, stratify=labels, seed=15)

	total_links = adj.count_nonzero()
	print('Backbone Model: '+ backbone_model)
	print('Total Nodes: '+str(adj.shape[0]))
	print('Total Links: '+str(total_links/2))
	print('# of Classes: '+str(nclass))

	total_features = features.shape[0]*features.shape[1]
	perturb_num_topology = int(sanitized_ratio_topology*total_links)
	perturb_num_feature = int(sanitized_ratio_feature*total_features)
	if modify_topology:
		print('Modify Topology by: '+adj_mode)
		print("Would perturb "+str(perturb_num_topology)+' edges')
	if modify_feature:
		print('Modify Feature by: '+feature_mode)
		print("Would perturb "+str(perturb_num_feature)+' features')
	# print("Would perturb "+str(perturb_num_topology)+' edges and '+str(perturb_num_feature)+' features.')

	if isinstance(adj, np.matrix):
		adj_tensor = torch.FloatTensor(adj)
	else:
		adj_tensor = sparse_mx_to_torch_tensor(adj)
	if isinstance(features, np.matrix):
		features_tensor = torch.FloatTensor(features)
	else:
		features_tensor = sparse_mx_to_torch_tensor(features)

	ones_adj = torch.ones_like(adj_tensor)
	ones_feature = torch.ones_like(features_tensor)

	adj_normalized_tensor = preprocess_adj_tensor(adj_tensor)

	for _ in tqdm(range(steps)):

		adj_difference_sum = torch.zeros_like(ones_adj)
		feature_difference_sum = torch.zeros_like(ones_feature)
		all_idx = list(idx_train)+list(idx_val)
		np.random.shuffle(all_idx)

		if isinstance(adj, np.matrix):
			adj_tensor = torch.FloatTensor(adj)
		else:
			adj_tensor = sparse_mx_to_torch_tensor(adj)
		if isinstance(features, np.matrix):
			features_tensor = torch.FloatTensor(features)
		else:
			features_tensor = sparse_mx_to_torch_tensor(features)

		for fold in range(num_fold):

			idx_val_tmp = all_idx[fold*len(all_idx)//num_fold:(fold+1)*len(all_idx)//num_fold]
			idx_train_tmp = list(set(all_idx)-set(idx_val_tmp))

			adj_difference, feature_difference = meta_gradient(adj, features, labels, idx_train_tmp, idx_val_tmp, modify_topology, modify_feature, backbone_model)

			if adj_mode == 'flip' or adj_mode == 'set':
				pass
			elif adj_mode == 'delete':
				adj_difference[adj_difference > 0] = 0 # only keep the negative terms
			elif adj_mode == 'add':
				adj_difference[adj_difference < 0] = 0 # only keep the positive terms

			if feature_mode == 'flip' or feature_mode == 'set':
				pass
			elif feature_mode == 'delete':
				feature_difference[feature_difference > 0] = 0 # only keep the negative terms
			elif feature_mode == 'add':
				feature_difference[feature_difference < 0] = 0 # only keep the positive terms

			adj_difference_sum += adj_difference
			feature_difference_sum += feature_difference

		topology_budget = int(perturb_num_topology//2*2//steps)
		feature_budget = int(perturb_num_feature//steps)

		if modify_topology:
			if adj_mode == 'set':
				matrix_norm = torch.sum(abs(adj_difference_sum), (0,1)).item()
				if matrix_norm == 0:
					lr = 1000
				else:
					lr = min(topology_budget/matrix_norm, 1000)
				adj += lr*adj_difference_sum.numpy()
				adj = adj.clip(min=0)
			else:
				S_adj = adj_difference_sum * (ones_adj - 2 * adj_tensor.data)
				tmp, idx = torch.topk(S_adj.flatten(), topology_budget)
				idx_row, idx_column = np.unravel_index(idx.numpy(), adj.shape)
				for i in range(len(idx_row)):
					if adj[idx_row[i], idx_column[i]] == 1:
						adj[idx_row[i], idx_column[i]] = 0
					else:
						adj[idx_row[i], idx_column[i]] = 1

		if modify_feature:
			if feature_mode == 'set':
				matrix_norm = torch.sum(abs(feature_difference_sum), (0,1)).item()
				if matrix_norm == 0:
					lr = 1000
				else:
					lr = min(feature_budget/matrix_norm, 1000)
				features += lr*feature_difference_sum.numpy()
				features = features.clip(min=0)
			else:
				S_feature = feature_difference_sum * (ones_feature - 2 * features_tensor.data)
				tmp, idx = torch.topk(S_feature.flatten(), feature_budget)
				idx_row, idx_column = np.unravel_index(idx.numpy(), features.shape)
				for i in range(len(idx_row)):
					if features[idx_row[i], idx_column[i]] == 1:
						features[idx_row[i], idx_column[i]] = 0
					else:
						features[idx_row[i], idx_column[i]] = 1

	features = sp.csr_matrix(features)
	# save the graph here
	modification_name = ''
	if modify_topology:
		modification_name += 'topology_'+adj_mode+'_'
	if modify_feature:
		modification_name += 'feature_'+feature_mode+'_'
	output_folder = output_data + '_' + modification_name + '.npz'
	np.savez(output_folder, adj_data=adj.data,
							adj_indices=adj.indices,
							adj_indptr=adj.indptr,
							adj_shape=adj.shape,
							attr_data=features.data,
							attr_indices=features.indices,
							attr_indptr=features.indptr,
							attr_shape=features.shape,
							labels=labels)
