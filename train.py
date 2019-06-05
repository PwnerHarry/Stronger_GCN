from __future__ import division
from __future__ import print_function
import time, os, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import load_data, accuracy
from models import graph_convolutional_network, linear_snowball, linear_tanh_snowball, snowball, truncated_krylov

parser = argparse.ArgumentParser()
# EXPERIMENT SETTINGS
parser.add_argument('--percent', type=float, default=0.05, help='Percentage of training set.')
parser.add_argument('--dataset', type=str, default='pubmed', help='Dataset (Cora, Citeseer, Pubmed)')
parser.add_argument('--public', type=int, default=0, help='Use the Public Setting of the Dataset of not')
parser.add_argument('--network', type=str, default='snowball', help='Network type (snowball, linear_snowball, linear_tanh_snowball, truncated_krylov)')
parser.add_argument('--validation', type=int, default=1, help='1 for turning on validation set, 0 for not')
# MODEL HYPERPARAMETERS
parser.add_argument('--lr', type=float, default=0.0076774, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.0062375, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=1, help='Number of hidden layers.')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')
parser.add_argument('--layers_factor', type=float, default=1, help='Factor for Determining the Number of Layers')
parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer')
parser.add_argument('--n_blocks', type=int, default=5, help='Number of Krylov blocks for truncated_krylov network')
# STOPPING CRITERIA
parser.add_argument('--epochs', type=int, default=3000, help='Number of max epochs to train.')
parser.add_argument('--consecutive', type=int, default= 200, help='Consecutive 100% training accuracy to stop')
parser.add_argument('--early_stopping', type=int, default= 100, help='Early Stopping')
parser.add_argument('--epochs_after_peak', type=int, default=200, help='Number of More Epochs Needed after 100% Training Accuracy Happens')
# MISC
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--runtimes', type=int, default=10, help='Runtimes.')
parser.add_argument('--debug', type=int, default = 1, help='1 for prompts during running, 0 for none')
parser.add_argument('--identifier', type=int, default=1234567, help='Identifier for the job')
parser.add_argument('--amp', type=int, default=1, help='1, 2 and 3 for NVIDIA apex amp optimization O1, O2 and O3, 0 for off')

args = parser.parse_args()

# Load data
dense_adj, features, labels = torch.load("%s_dense_adj.pt" % args.dataset), torch.load("%s_features.pt"%args.dataset), torch.load("%s_labels.pt"%args.dataset)
indices = torch.nonzero(dense_adj).t(); values = dense_adj[indices[0], indices[1]]
adj = torch.sparse.FloatTensor(indices, values, dense_adj.size()).clone()
del dense_adj, indices, values

# set environment
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.cuda = torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
if args.amp:
    try:
        from apex import amp
    except ModuleNotFoundError:
        args.amp = 0
        print('module apex not found, amp (mixed-precision acceleration) disabled')

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])# - args.regularization_factor * regularizer
    acc_train = accuracy(output[idx_train], labels_train)
    if args.amp:
        with amp.scale_loss(loss_train, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)
    acc_val = accuracy(output[idx_val], labels_val)
    if args.debug:
        print('E%04d, loss_train: %4.2e, acc_train: %6.2f%%, best_val: %5.2f%%, best_test: %5.2f%%' % (epoch + 1, loss_train.item(), 100 * acc_train.item(), best_val, 100 * best_test), end = " ")
    return 100 * acc_train.item(), loss_train.item(), 100 * acc_val.item()

def test():
    model.eval(); output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels_test)
    acc_test = accuracy(output[idx_test], labels_test)
    if args.debug:
        print("loss_test: %4.2e, acc_test: %5.2f%%" % (loss_test.item(), 100 * acc_test.item()), end = " ")
    return acc_test

def layer_numbers(idx_train, adj, percent, factor):
    # DEPRECATED
    local_adj = adj.clone().cpu(); local_idx_train = idx_train.clone().cpu()
    s = np.zeros((local_idx_train.shape[0], adj.shape[0]))
    s[np.arange(local_idx_train.shape[0]), local_idx_train] = 1
    local_s = torch.Tensor(s).cpu()
    j = -1
    redundant_addition=0
    new_addition=0
    while (new_addition>factor*percent*redundant_addition) | (redundant_addition==0):#(sum(np.sum(np.transpose(torch.spmm(adj,torch.transpose(s,0,1)).numpy())>0,0)>0) - np.sum(np.sum((s>0).numpy(),0)>0) )>0:
        j=j+1
        reached_nodes = (np.sum((local_s>0).numpy(),0)>0)*1
        new_reach_nodes = (np.sum(np.transpose(torch.spmm(local_adj,torch.transpose(local_s,0,1)).numpy())>0,0)>0)*1 -(reached_nodes)*1
        
        addition = np.sum(np.transpose(torch.spmm(local_adj,torch.transpose(local_s,0,1)).numpy())>0,0) - np.sum((local_s>0).numpy(),0)
        
        redundant_addition = np.dot(addition,reached_nodes)
        new_addition = np.dot(addition, new_reach_nodes)
        
        local_s = torch.transpose(torch.spmm(local_adj,torch.transpose(local_s,0,1)),0,1)
    return j

# setup training, validation and testing set if public
if args.public == 1:
    if args.dataset == 'cora':
        idx_train, idx_val, idx_test = range(140), range(140, 640), range(1708, 2708)
        percent = 140 / 2708
    elif args.dataset == 'citeseer':
        idx_train, idx_val, idx_test = range(120), range(120, 620), range(2312, 3312)
        percent = 120 / 3312
    elif args.dataset == 'pubmed':
        idx_train, idx_val, idx_test = range(60), range(60, 560), range(18717, 19717)
        percent = 60 / 19717
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    if args.cuda:
        idx_train, idx_val, idx_test = torch.LongTensor(idx_train).cuda(), torch.LongTensor(idx_val).cuda(), torch.LongTensor(idx_test).cuda()
        labels_train, labels_val, labels_test = labels_train.cuda(), labels_val.cuda(), labels_test.cuda()

if args.layers == 0:
    percent = args.percent
    layer_num = layer_numbers(idx_train,adj,percent,args.layers_factor)
else:
    layer_num = args.layers

if args.activation == 'identity':
    activation = lambda X: X
elif args.activation == 'tanh':
    activation = torch.tanh
else:
    activation = eval("F.%s" % args.activation)

Network = eval(args.network)
if args.network == 'snowball':
    model = Network(nfeat=features.shape[1], nlayers=layer_num, nhid=args.hidden, nclass=labels.max().item() + 1,
                dropout=args.dropout, activation = activation)
if args.network == 'snowball_noclassifier':
    model = Network(nfeat=features.shape[1], nlayers=layer_num, nhid=args.hidden, nclass=labels.max().item() + 1,
                dropout=args.dropout, activation = activation)
elif args.network == 'linear_snowball':
    model = Network(nfeat=features.shape[1], nlayers=layer_num, nhid=args.hidden, nclass=labels.max().item() + 1,
                    dropout=args.dropout)
elif args.network == 'linear_tanh_snowball':
    model = Network(nfeat=features.shape[1], nlayers=layer_num, nhid=args.hidden, nclass=labels.max().item() + 1,
                    dropout=args.dropout)
elif args.network == 'truncated_krylov':
    ADJ_EXPONENTIALS, accumulated_exponential = [], torch.eye(adj.size()[0])
    if args.cuda:
        accumulated_exponential = accumulated_exponential.cuda()
    for i in range(args.n_blocks):
        ADJ_EXPONENTIALS.append(accumulated_exponential)
        accumulated_exponential = torch.spmm(adj, accumulated_exponential)
    del accumulated_exponential
    if not args.amp:
        for i in range(args.n_blocks):
            dense_exponent = ADJ_EXPONENTIALS[i]
            indices = torch.nonzero(dense_exponent).t(); values = dense_exponent[indices[0], indices[1]]
            ADJ_EXPONENTIALS[i] = torch.sparse.FloatTensor(indices, values, dense_exponent.size())
    model = Network(nfeat=features.shape[1], nlayers=layer_num, nhid=args.hidden, nclass=labels.max().item() + 1,
                dropout=args.dropout, activation = activation, n_blocks = args.n_blocks, ADJ_EXPONENTIALS = ADJ_EXPONENTIALS)

# set optimizer
if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# send to GPU
if args.cuda:
    model.cuda()
if args.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O%d" % args.amp)
    adj = adj.to_dense()

# experiment
result3, total_running_time = [], 0
for runtime in range(args.runtimes):
    model.reset_parameters()
    if args.debug:
        best_val, best_test = 0, 0
    if args.public != 1:
        if args.public == 2:
            all_data = np.arange(adj.shape[0]).astype(int)
            idx_train, idx_val, idx_test = [], [], []
            all_class = np.unique(labels.cpu().numpy())
            for c in all_class:
                idx_train = np.hstack([idx_train,random.sample(list(np.where(labels.cpu().numpy()==c)[0].astype(int)), 20)])
            others = np.delete(all_data.astype(int), idx_train.astype(int))
            for c in all_class:
                idx_val = np.hstack([idx_val,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), int(500/all_class.shape[0]) )])
            others = np.delete(others.astype(int), idx_val.astype(int))
            for c in all_class:
                idx_test = np.hstack([idx_test,random.sample(list(np.where(labels[others].cpu().numpy()==c)[0].astype(int)), min(int(1000/all_class.shape[0]), np.where(labels[others].cpu().numpy()==c)[0].astype(int).shape[0]) )])
        else:
            all_data = np.arange(adj.shape[0]).astype(int)
            idx_train = []
            for c in np.unique(labels.cpu().numpy()):
                idx_train = np.hstack([idx_train,random.sample(list(np.where(labels.cpu().numpy()==c)[0].astype(int)), int(np.where(labels.cpu().numpy()==c)[0].shape[0]*args.percent)+1)])
            others = np.delete(all_data.astype(int), idx_train.astype(int))
            random.shuffle(others)
            idx_val, idx_test = others[0:500], others[500:1500] 
        labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
        if args.cuda:
            idx_train, idx_val, idx_test = torch.LongTensor(idx_train).cuda(), torch.LongTensor(idx_val).cuda(), torch.LongTensor(idx_test).cuda()
            labels_train, labels_val, labels_test = labels_train.cuda(), labels_val.cuda(), labels_test.cuda()

    t_total = time.time()
    early_stopping, consecutive = 0, 0
    epoch = 0
    peaked = False
    best_train, test_best_val, best_validation = 0, 0, 0
    while epoch <= args.epochs:
        if args.debug:
            print("R%02d" % runtime, end = " ")
        acc_train, train_loss, acc_val = train(epoch)
        if args.validation or args.debug:
            if acc_val >= best_validation:
                best_validation = acc_val
                test_best_val = test().cpu().numpy()
                acc_test = test_best_val
                early_stopping = 0
                if args.debug:
                    print('test_best_val: %.2f%%' % (100 * float(test_best_val)), end = "")
            elif args.debug:
                acc_test = test().cpu().numpy()
        if args.validation == 0 and acc_train >= best_train:
            best_train = acc_train
            early_stopping = 0
        else:
            early_stopping += 1
            
        if early_stopping >= args.early_stopping: break
        if abs(acc_train - 100) < 1e-2:
            if consecutive == 0:
                if args.debug:
                    print('first peak met at epoch %d' % epoch)
                args.epochs = epoch + args.epochs_after_peak
            consecutive += 1
        if consecutive >= args.consecutive: break
        if args.debug:
            best_val, best_test = max(best_val, acc_val), max(best_test, acc_test)
        epoch += 1
        if args.debug:
            print("", end = "\n")
    if args.validation == 1:
        runtime_result = test_best_val
    else:
        runtime_result = test().cpu().numpy()
    if args.debug:
        print("R%d finished with %.2fs elapsed, acc: %5.2f%%" % (runtime, time.time() - t_total, 100 * float(runtime_result)), end = "\n")
        print("best_val %.2f%%, best_test %.2f%%, test_best_val %.2f%%" % (best_val, 100 * best_test, 100 * test_best_val), end = "\n")
    total_running_time = total_running_time + time.time() - t_total
    result3.append(runtime_result)
    if args.debug:
        print("", end = "\n")
    if runtime_result < 0.4: # don't bother if acc < 0.4!
        break
if args.debug:
    print("mean result: ", np.mean(result3), "total running time: ", total_running_time, "All results: ", result3)
else:
    print(np.mean(result3))
    script = open("%d.txt" % args.identifier, 'w'); script.write("%e" % np.mean(result3)); script.close()
