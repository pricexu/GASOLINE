import os
import sys
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.model_selection import train_test_split

# from numba import jit

def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def bisection(a,eps,xi,ub=1):
	pa = torch.clamp(a, 0, ub)
	if torch.sum(pa) <= eps:
		print('np.sum(pa) <= eps !!!!')
		upper_S_update = pa
	else:
		mu_l = torch.min(a-1)
		mu_u = torch.max(a)
		mu_a = (mu_u + mu_l)/2
		while torch.abs(mu_u - mu_l)>xi:
			mu_a = (mu_u + mu_l)/2
			gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
			gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
			if gu == 0:
				print('gu == 0 !!!!!')
				break
			if torch.sign(gu) == torch.sign(gu_l):
				mu_l = mu_a
			else:
				mu_u = mu_a
		upper_S_update = torch.clamp(a-mu_a, 0, ub)
	return upper_S_update

def load_npz(file_name, is_sparse=True):
	if not file_name.endswith('.npz'):
		file_name += '.npz'

	with np.load(file_name) as loader:
		if is_sparse:
			adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
										loader['adj_indptr']), shape=loader['adj_shape'])

			if 'attr_data' in loader:
				features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
											 loader['attr_indptr']), shape=loader['attr_shape'])
			else:
				print('no sparse feature!')
				features = None

			labels = loader.get('labels')

		else:
			adj = loader['adj_data']

			if 'attr_data' in loader:
				features = loader['attr_data']
			else:
				features = None

			labels = loader.get('labels')

	return adj, features, labels

def get_info(dataset, require_lcc=True):
	_A_obs, _X_obs, _z_obs = load_npz(dataset)
	_A_obs = _A_obs + _A_obs.T
	_A_obs = _A_obs.tolil()
	_A_obs[_A_obs > 1] = 1

	if _X_obs is None:
		_X_obs = sp.csr_matrix(np.eye(_A_obs.shape[0]), dtype=np.float32)

	if require_lcc:
		lcc = largest_connected_components(_A_obs)

		_A_obs = _A_obs[lcc][:,lcc]
		_X_obs = _X_obs[lcc]
		_z_obs = _z_obs[lcc]

		assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

	_A_obs.setdiag(0)
	_A_obs = _A_obs.astype("float32").tocsr()
	_A_obs.eliminate_zeros()

	assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
	assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

	return _A_obs, _X_obs, _z_obs

def largest_connected_components(adj, n_components=1):
	"""Select the largest connected components in the graph.
	Parameters
	"""
	_, component_indices = sp.csgraph.connected_components(adj)
	component_sizes = np.bincount(component_indices)
	components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
	nodes_to_keep = [
		idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
	# print("Selecting {0} largest connected components".format(n_components))
	return nodes_to_keep



def load_data(dataset="cora"):

	print('Loading {} dataset...'.format(dataset))
	adj, features, labels = get_info(dataset)

	return adj, features, labels

def load_adj(dataset="cora"):

	print('Loading {} adj...'.format(dataset))
	with np.load(dataset) as loader:
		adj = sp.csr_matrix((loader['data'], loader['indices'],
									loader['indptr']), shape=loader['shape'])

	return adj

def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
	"""This setting follows nettack/mettack, where we split the nodes
	into 10% training, 10% validation and 80% testing data

	Parameters
	----------
	nnodes : int
		number of nodes in total
	val_size : float
		size of validation set
	test_size : float
		size of test set
	stratify :
		data is expected to split in a stratified fashion. So stratify should be labels.
	seed : int or None
		random seed

	Returns
	-------
	idx_train :
		node training indices
	idx_val :
		node validation indices
	idx_test :
		node test indices
	"""

	assert stratify is not None, 'stratify cannot be None!'

	if seed is not None:
		np.random.seed(seed)

	idx = np.arange(nnodes)
	train_size = 1 - val_size - test_size
	idx_train_and_val, idx_test = train_test_split(idx,
												   random_state=None,
												   train_size=train_size + val_size,
												   test_size=test_size,
												   stratify=stratify)

	if stratify is not None:
		stratify = stratify[idx_train_and_val]

	idx_train, idx_val = train_test_split(idx_train_and_val,
										  random_state=None,
										  train_size=(train_size / (train_size + val_size)),
										  test_size=(val_size / (train_size + val_size)),
										  stratify=stratify)

	return idx_train, idx_val, idx_test

def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)

	return features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch tensor."""
	return torch.FloatTensor(sparse_mx.astype(np.float32).toarray())

def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return adj_normalized

def normalize_adj_tensor(adj):
	"""Symmetrically normalize adjacency tensor."""
	rowsum = torch.sum(adj,1)
	d_inv_sqrt = torch.pow(rowsum, -0.5)
	d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
	d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
	return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def preprocess_adj_tensor(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj_tensor(adj + torch.eye(adj.shape[0]))
	return adj_normalized

def cross_entropy(pred, soft_targets):
	logsoftmax = nn.LogSoftmax(dim=1)
	return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
