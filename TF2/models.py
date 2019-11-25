from layers import LAYER_SNOWBALL, LAYER_TRUNCATED_KRYLOV
import numpy as np, tensorflow as tf
from utils import dense2sparse

class GCN(tf.keras.Model):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, name=None, dtype=tf.float32):
        super(GCN, self).__init__(name=name)
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = []

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()
    
    def vars_trainable(self):
        list_vars_trainable = []
        for layer in self.hidden:
            list_vars_trainable.extend(layer.vars_trainable())
        list_vars_trainable.extend(self.out.vars_trainable())
        return list_vars_trainable

class snowball(GCN):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, name=None, dtype=tf.float32):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout, name=name)
        self.activation = activation
        for k in range(nlayers):
            layer = LAYER_SNOWBALL(k * nhid + nfeat, nhid, dtype=dtype)
            self.hidden.append(layer)
        self.out = LAYER_SNOWBALL(nlayers * nhid + nfeat, nclass, dtype=dtype)
    
    def __call__(self, x, adj, train=True):
        if train:
            dropout = tf.nn.dropout
        else:
            dropout = tf.identity
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(dropout(self.activation(layer(x, adj)), self.dropout))
            else:
                list_output_blocks.append(dropout(self.activation(layer(tf.concat([x] + list_output_blocks[0: layer_num], 1), adj)), self.dropout))
        output = self.out(tf.concat([x] + list_output_blocks, 1), adj, eye=False)
        return tf.nn.softmax(output, axis=1)

class truncated_krylov(GCN):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features, name=None, dtype=tf.float32):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout, name=name)
        self.activation = activation
        LIST_A_EXP, LIST_A_EXP_X, A_EXP = [], [], tf.eye(tf.shape(adj)[0], dtype=dtype)
        for _ in range(n_blocks):
            if nlayers > 1: LIST_A_EXP.append(dense2sparse(A_EXP))
            LIST_A_EXP_X.append(tf.matmul(A_EXP, features))
            A_EXP = tf.matmul(tf.sparse.to_dense(adj), A_EXP)
        layer = LAYER_TRUNCATED_KRYLOV(nfeat, nhid, n_blocks, LIST_A_EXP_X_CAT=tf.concat(LIST_A_EXP_X, 1), dtype=dtype)
        self.hidden.append(layer)
        for _ in range(nlayers - 1):
            layer = LAYER_TRUNCATED_KRYLOV(nhid, nhid, n_blocks, LIST_A_EXP=LIST_A_EXP, dtype=dtype)
            self.hidden.append(layer)
        self.out = LAYER_TRUNCATED_KRYLOV(nhid, nclass, 1, dtype=dtype)

    def __call__(self, x, adj, train=True):
        if train:
            dropout = tf.nn.dropout
        else:
            dropout = tf.identity
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(dropout(self.activation(layer(x, adj)), self.dropout))
            else:
                list_output_blocks.append(dropout(self.activation(layer(list_output_blocks[layer_num - 1], adj)), self.dropout))
        output = self.out(list_output_blocks[self.nlayers - 1], adj, eye=True)
        return tf.nn.softmax(output, axis=1)