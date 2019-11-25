from __future__ import division, print_function
import time, os, argparse, random, torch
import numpy as np
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from utils import accuracy, load_dataset, split_dataset, train
from models import snowball, truncated_krylov

parser = argparse.ArgumentParser()
# EXPERIMENT SETTINGS
parser.add_argument('--dataset', type=str, default='pubmed', help='Dataset (cora, citeseer, pubmed)')
parser.add_argument('--public', type=int, default=0, help='Use the Public Setting of the Dataset of not')
parser.add_argument('--percent', type=float, default=0.0003, help='Percentage of training set.')
parser.add_argument('--network', type=str, default='linear_snowball', help='Network type (snowball, linear_snowball, truncated_krylov)')
parser.add_argument('--validation', type=int, default=1, help='1 for tuning on validation set, 0 for not')
# MODEL HYPERPARAMETERS
parser.add_argument('--lr', type=float, default=4.7041e-05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.033218, help='Weight decay.')
parser.add_argument('--dropout', type=float, default=0.19567, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=200, help='Width of hidden layers')
parser.add_argument('--layers', type=int, default=8, help='Number of hidden layers, i.e. network depth')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')
parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer')
parser.add_argument('--n_blocks', type=int, default=5, help='Number of Krylov blocks for truncated_krylov network')
# STOPPING CRITERIA
parser.add_argument('--epochs', type=int, default=3000, help='Number of max epochs to train.')
parser.add_argument('--consecutive', type=int, default= 200, help='Consecutive 100% training accuracy to stop')
parser.add_argument('--early_stopping', type=int, default= 100, help='Early Stopping')
parser.add_argument('--epochs_after_peak', type=int, default=200, help='Number of More Epochs Needed after 100% Training Accuracy Happens')
# MISC
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--walltime', type=float, default=10800, help='Random seed.')
parser.add_argument('--runtimes', type=int, default=10, help='Runtimes.')
parser.add_argument('--debug', type=int, default=1, help='1 for prompts during running, 0 for none')
parser.add_argument('--identifier', type=int, default=1234567, help='Identifier for the job')
# FOR TORCH IMPLEMENTATION
parser.add_argument('--amp', type=int, default=2, help='1, 2 and 3 for NVIDIA apex amp optimization O1, O2 and O3, 0 for off')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.cuda = torch.cuda.is_available()
adj, features, labels = load_dataset(args.dataset)
features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()

if args.amp: # amp will probably require weight_decay in optimizer to stabilize
    try:
        from apex import amp
        adj = adj.to_dense() # In PyTorch 1.3, torch.sparse.mm(half, half) is still not implemented!!
    except ModuleNotFoundError:
        args.amp = 0
        print('module apex not found, mixed-precision acceleration disabled')

if args.activation == 'identity' or args.network == 'linear_snowball':
    activation = lambda X: X
elif args.activation == 'tanh':
    activation = torch.tanh
else:
    activation = eval("F.%s" % args.activation)

# EXPERIMENT
result3, total_running_time = [], 0
for runtime in range(args.runtimes):
    if runtime and total_running_time * (runtime + 1.2) / runtime > args.walltime: break
    if args.network == 'snowball' or args.network == 'linear_snowball':
        model = snowball(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout, activation=activation)
    elif args.network == 'truncated_krylov':
        if args.amp == 2:
            adj_feed, features_feed = adj.half(), features.half()
        else:
            adj_feed, features_feed = adj, features
        model = truncated_krylov(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout, activation = activation, n_blocks=args.n_blocks, adj=adj_feed, features=features_feed)
    class_optimizer = eval('optim.%s' % args.optimizer)
    if args.cuda: model.cuda()
    optimizer = class_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.amp: model, optimizer = amp.initialize(model, optimizer, opt_level="O%d" % args.amp)
    idx_train, idx_val, idx_test, labels_train, labels_val, labels_test = split_dataset(adj.shape[0], labels, args.dataset, args.public, args.percent, args.cuda)
    epoch, test_best_train, best_train, test_best_val, best_validation = 0, 0, 0, 0, 0
    if args.debug: best_val, best_test = 0, 0
    early_stopping, consecutive, peaked = 0, 0, False
    t_total = time.time()
    while epoch <= args.epochs:
        acc_train, loss_train = train(model, optimizer, features, adj, labels, idx_train, idx_val, args.amp)
        model.eval()
        output = model(features, adj)
        output_train, output_test = output[idx_train], output[idx_test]
        acc_train, acc_test = accuracy(output[idx_train], labels_train).item(), accuracy(output[idx_test], labels_test).item()
        if args.validation or args.debug:
            output_val = output[idx_val]
            acc_val = accuracy(output[idx_val], labels_val).item()
            if acc_val >= best_validation:
                best_validation = acc_val
                test_best_val = acc_test
                early_stopping = 0
        if args.validation == 0 and acc_train >= best_train:
            best_train = acc_train
            early_stopping = 0
            test_best_train = acc_test
        else:
            early_stopping += 1
        if early_stopping >= args.early_stopping: break
        if abs(acc_train - 1) < 1e-2:
            if consecutive == 0:
                args.epochs = epoch + args.epochs_after_peak
                if args.debug: print('first peak met at epoch %d' % epoch)
            consecutive += 1
        if consecutive >= args.consecutive: break
        if args.debug: best_val, best_test = max(best_val, acc_val), max(best_test, acc_test)
        epoch += 1
        if args.debug: print("\r\rR%02d " % runtime, 'E%04d, loss_train: %4.2e, acc_train: %6.2f%%, best_val: %5.2f%%, best_test: %5.2f%%, test_best_val: %.2f%%' % (epoch + 1, loss_train, 100 * acc_train, 100 * best_val, 100 * best_test, 100 * float(test_best_val)), end="")
    if args.validation == 1:
        acc_run = test_best_val
    else:
        acc_run = test_best_train
    if args.debug:
        print("R%d finished with %.2fs elapsed, acc: %5.2f%%" % (runtime, time.time() - t_total, 100 * float(acc_run)), end="\n")
        print("best_val %.2f%%, best_test %.2f%%, test_best_val %.2f%%" % (best_val, 100 * best_test, 100 * test_best_val), end="\n")
    total_running_time = total_running_time + time.time() - t_total
    result3.append(acc_run)
    if args.debug: print("", end="\n")
    del model, optimizer
    if args.cuda: torch.cuda.empty_cache()
print("mean result: ", np.mean(result3), "total running time: ", total_running_time, "All results: ", result3)
script = open("%d.txt" % args.identifier, 'w'); script.write("%e" % np.mean(result3)); script.close()