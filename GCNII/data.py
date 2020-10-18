import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import ShuffleSplit
from utils import sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
import pickle as pkl
import sys
import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
from utils import parse_index_file, normalize

def full_citation(dataset_str="cora"):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i, name in enumerate(names):
        with open("dataset/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            objects.append(pkl.load(f, encoding="latin1"))    

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # For this dataset, there are some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder))
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))

    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = full_citation(dataset_str="cora")
    print(adj.shape, features.shape, idx_train.shape)