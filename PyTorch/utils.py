import numpy as np
import scipy.sparse as sp
import torch, random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, math
try:
    from apex import amp
    global_flag_amp = True
except ModuleNotFoundError:
    global_flag_amp = False

def train(model, optimizer, features, adj, labels, idx_train, idx_val, flag_amp):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    labels_train, output_train = labels[idx_train], output[idx_train]
    loss_train = F.nll_loss(output_train, labels_train)
    acc_train = accuracy(output_train, labels_train)
    if global_flag_amp and flag_amp:
        with amp.scale_loss(loss_train, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_train.backward()
    optimizer.step()
    return 100 * acc_train.item(), loss_train.item()

# def test(model, features, adj, idx_test, labels_test, flag_debug):
#     model.eval()
#     output = model(features, adj)
#     acc_test = accuracy(output[idx_test], labels_test)
#     if flag_debug: print("loss_test: %4.2e, acc_test: %5.2f%%" % (F.nll_loss(output[idx_test], labels_test).item(), 100 * acc_test.item()), end=" ")
#     return acc_test

def split_dataset(num_nodes, labels, dataset, public, percent, flag_cuda):
    if public == 1:
        if dataset == 'cora':
            idx_train, idx_val, idx_test = range(140), range(140, 640), range(1708, 2708)
        elif dataset == 'citeseer':
            idx_train, idx_val, idx_test = range(120), range(120, 620), range(2312, 3312)
        elif dataset == 'pubmed':
            idx_train, idx_val, idx_test = range(60), range(60, 560), range(18717, 19717)
    elif public == 2:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels.cpu().numpy())
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train,random.sample(list(np.where(labels.cpu().numpy()==c)[0].astype(int)), 20)])
        others = np.delete(all_data.astype(int), idx_train.astype(int))
        for c in all_class:
            idx_val = np.hstack([idx_val,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), math.ceil(500/all_class.shape[0]) )])
        others = np.delete(others.astype(int), idx_val.astype(int))
        for c in all_class:
            idx_test = np.hstack([idx_test,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), min(math.ceil(1000/all_class.shape[0]), np.where(labels[others].cpu().numpy()==c)[0].astype(int).shape[0]))])
    elif public == 0:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels.cpu().numpy())
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train,random.sample(list(np.where(labels.cpu().numpy()==c)[0].astype(int)), math.ceil(np.where(labels.cpu().numpy()==c)[0].shape[0]*percent))])
        others = np.delete(all_data.astype(int), idx_train.astype(int))
        for c in all_class:
            idx_val = np.hstack([idx_val,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), math.ceil(500/all_class.shape[0]) )])
        others = np.delete(others.astype(int), idx_val.astype(int))
        for c in all_class:
            idx_test = np.hstack([idx_test,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), min(math.ceil(1000/all_class.shape[0]), np.where(labels[others].cpu().numpy()==c)[0].astype(int).shape[0]))])
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    if flag_cuda:
        idx_train, idx_val, idx_test = torch.LongTensor(idx_train).cuda(), torch.LongTensor(idx_val).cuda(), torch.LongTensor(idx_test).cuda()
        labels_train, labels_val, labels_test = labels_train.cuda(), labels_val.cuda(), labels_test.cuda()
    return idx_train, idx_val, idx_test, labels_train, labels_val, labels_test

def load_dataset(dataset):
    dense_adj, features, labels = torch.load("%s_dense_adj.pt" % dataset), torch.load("%s_features.pt" % dataset), torch.load("%s_labels.pt" % dataset)
    indices = torch.nonzero(dense_adj).t(); values = dense_adj[indices[0], indices[1]]
    adj = torch.sparse.FloatTensor(indices, values, dense_adj.size()).clone()
    del dense_adj, indices, values
    return adj, features, labels

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset_str))
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

    adj=sp.coo_matrix(adj,dtype=np.float32)
    # build graph


    # build symmetric adjacency matrix

    features = normalize(features)
    adj =  normalize(sp.eye(adj.shape[0]) + adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double() #.average()????
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
