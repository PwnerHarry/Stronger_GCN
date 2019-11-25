import sys, pickle as pkl, numpy as np
import scipy.sparse as sp
import networkx as nx

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    print('Loading %s dataset...' % (dataset_str))
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.csr_matrix(features, dtype=np.float32)[list(np.where(np.sum(labels,1)==1)[0]),:]
    adj = sp.csr_matrix(adj, dtype=np.float32)[:,list(np.where(np.sum(labels,1)==1)[0])][list(np.where(np.sum(labels,1)==1)[0]),:]
    adj = sp.coo_matrix(adj,dtype=np.float32)
    features = normalize(features)
    adj =  normalize(sp.eye(adj.shape[0]) + adj).astype(np.float32)
    features = np.array(features.todense())
    labels = np.where(labels)[1]
    return adj, features, labels
    
for dataset in ['cora', 'citeseer', 'pubmed']:
    adj, features, labels = load_data(dataset)
    sp.save_npz("%s_adj.npz" % dataset, adj)
    np.save("%s_features.npy" % dataset, features)
    np.save("%s_labels.npy" % dataset, labels)

