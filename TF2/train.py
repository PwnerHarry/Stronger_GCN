from __future__ import division, print_function
import tensorflow as tf, numpy as np
import time, os, argparse, random, datetime
from utils import load_dataset, split_dataset, trainer, Adam, RMSprop
from models import snowball, truncated_krylov

parser = argparse.ArgumentParser()
# EXPERIMENT SETTINGS
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset (cora, citeseer, pubmed)')
parser.add_argument('--public', type=int, default=1, help='Use the Public Setting of the Dataset of not')
parser.add_argument('--percent', type=float, default=0, help='Percentage of training set.')
parser.add_argument('--network', type=str, default='linear_snowball', help='Network type (snowball, linear_snowball, truncated_krylov)')
parser.add_argument('--validation', type=int, default=1, help='1 for tuning on validation set, 0 for not')
# MODEL HYPERPARAMETERS
parser.add_argument('--lr', type=float, default=0.0017077, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.033218, help='Weight decay.')
parser.add_argument('--dropout', type=float, default=0.95232, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=2100, help='Width of hidden layers')
parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers, i.e. network depth')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')
parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer')
parser.add_argument('--n_blocks', type=int, default=10, help='Number of Krylov blocks for truncated_krylov network')
# STOPPING CRITERIA
parser.add_argument('--epochs', type=int, default=5000, help='Number of max epochs to train.')
parser.add_argument('--early_stopping', type=int, default=50, help='Early Stopping')
# MISC
parser.add_argument('--amp', type=int, default=1, help='Auto Mixed Precision')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--walltime', type=float, default=10800, help='Random seed.')
parser.add_argument('--runtimes', type=int, default=10, help='Runtimes.')
parser.add_argument('--debug', type=int, default=1, help='1 for prompts during running, 0 for none')
parser.add_argument('--identifier', type=int, default=1234567, help='Identifier for the job')
# parser.add_argument('--precision', type=str, default='single', help='precision for floats')
args = parser.parse_args()
DTYPE = tf.float32

if args.network == 'linear_snowball' or args.activation == 'identity':
    activation = tf.identity
elif args.activation == 'relu':
    activation = tf.nn.relu
else:
    activation = eval("tf.math.%s" % args.activation)

np.random.seed(args.seed)
adj, features, labels = load_dataset(args.dataset, DTYPE)

if args.network == 'linear_snowball' or args.network == 'snowball':
    model = eval('snowball')(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=int(tf.reduce_max(labels)) + 1, dropout=args.dropout, activation=activation, dtype=DTYPE)
elif args.network == 'truncated_krylov':
    model = eval('truncated_krylov')(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=int(tf.reduce_max(labels)) + 1, dropout=args.dropout, activation=activation, n_blocks=args.n_blocks, adj=adj, features=features, dtype=DTYPE)

tracker_loss_train = tf.keras.losses.SparseCategoricalCrossentropy(name='loss_train')
tracker_loss_val = tf.keras.losses.SparseCategoricalCrossentropy(name='loss_val')
tracker_loss_test = tf.keras.losses.SparseCategoricalCrossentropy(name='loss_test')
tracker_acc_train = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_train')
tracker_acc_val = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_val')
tracker_acc_test = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_test')

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = str(args.amp)

acc_runs, time_total = [], 0
for runtime in range(args.runtimes):
    if runtime and time_total * (runtime + 1.2) / runtime > args.walltime: break
    if args.optimizer == 'RMSprop':
        optimizer = RMSprop(lr=args.lr, weight_decay=args.weight_decay, rho=0.99, epsilon=1e-8) # setting eps=1e-8 to get closer to PyTorch!
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr, weight_decay=args.weight_decay, epsilon=1e-8)
    if args.amp: optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer) 
    idx_train, idx_val, idx_test, labels_train, labels_val, labels_test = split_dataset(adj.shape[0], labels, args.dataset, args.public, args.percent)
    epoch, early_stopping, acc_best_train, acc_test_best_train, acc_test_best_val, best_train, best_val, best_test = 0, 0, 0, 0, 0, float('Inf'), float('Inf'), float('Inf')
    train = trainer()
    if args.debug:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        run_log_dir = 'logs/gradient_tape/' + current_time + '/run'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        run_summary_writer = tf.summary.create_file_writer(run_log_dir)
    time_run_start = time.time()
    for epoch in range(args.epochs):
        loss_train = train(model, optimizer, features, adj, labels, idx_train, tracker_loss_train)
        output = model(features, adj, train=False)
        output_test = tf.gather(output, idx_test)
        output_train = tf.gather(output, idx_train)
        tracker_acc_train(labels_train, output_train)
        tracker_acc_test(labels_test, output_test)
        acc_best_train = max(acc_best_train, tracker_acc_train.result())
        if loss_train < best_train:
            best_train, acc_test_best_train = loss_train, tracker_acc_test.result()
            if tracker_acc_train.result() == 1: early_stopping = 0
        if args.validation or args.debug:
            labels_val, output_val = tf.gather(labels, idx_val), tf.gather(output, idx_val)
            acc_val, loss_val = tracker_acc_val(labels_val, output_val), tracker_loss_val(labels_val, output_val)
            if loss_val < best_val:
                best_val, acc_test_best_val, early_stopping = loss_val, tracker_acc_test.result(), 0
        if tracker_acc_train.result() < acc_best_train and loss_train > best_train:
            if args.validation:
                if loss_val > best_val:
                    early_stopping += 1
                else:
                    early_stopping = 0
            else:
                early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping >= args.early_stopping: break
        if args.debug:
            loss_test = tracker_loss_test(labels_test, output_test)
            best_test = min(best_test, loss_test)
            if args.validation == 1:
                acc_run = acc_test_best_val
            else:
                acc_run = acc_test_best_train
            print('R%02d E%04d l_train: %4.2e a_train: %5.1f%% a_val %5.1f%% a_test %5.1f%% ba_train %4.2e ba_val %4.2e ba_test %4.2e a_run %5.1f%% stop %d' % (runtime, epoch + 1, loss_train, 100 * tracker_acc_train.result(), 100 * acc_val, 100 * tracker_acc_test.result(), best_train, best_val, best_test, 100 * acc_run, args.early_stopping - early_stopping))
            with train_summary_writer.as_default():
                tf.summary.scalar('accuracy', tracker_acc_train.result(), step=epoch)
                tf.summary.scalar('loss', loss_train, step=epoch)
            with val_summary_writer.as_default():
                tf.summary.scalar('accuracy', tracker_acc_val.result(), step=epoch)
                tf.summary.scalar('loss', loss_val, step=epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('accuracy', tracker_acc_test.result(), step=epoch)
                tf.summary.scalar('loss', loss_test, step=epoch)
            with run_summary_writer.as_default():
                tf.summary.scalar('accuracy', acc_run, step=epoch)
        epoch += 1
        tracker_acc_train.reset_states(); tracker_acc_val.reset_states(); tracker_acc_test.reset_states()
    time_run = time.time() - time_run_start
    time_total = time_total + time_run
    if args.validation == 1:
        acc_run = acc_test_best_val
    else:
        acc_run = acc_test_best_train
    acc_runs.append(acc_run)
    print("R%d finished with %d epochs in %.2fs (%.2es per epoch), acc_run: %5.1f%%" % (runtime, epoch + 1, time_run, time_run / (epoch + 1), 100 * float(acc_run)))
    model.reset_parameters()
print("mean result: ", np.mean(acc_runs), "total time elapsed: %.2fs" % time_total, "all results: ", acc_runs)
script = open("%d.txt" % args.identifier, 'w'); script.write("%e" % np.mean(acc_runs)); script.close()