import math, tensorflow as tf

class LAYER_GENERAL(tf.Module):
    def __init__(self, name=None):
        super(LAYER_GENERAL, self).__init__(name=name)
    def vars_trainable(self):
        return [self.weight, self.bias]

class LAYER_SNOWBALL(LAYER_GENERAL):
    def __init__(self, features_in, features_out, name=None, dtype=tf.float32):
        super(LAYER_SNOWBALL, self).__init__(name=name)
        self.features_in, self.features_out = features_in, features_out
        self.dtype = dtype
        stdv_weight, stdv_bias = 1. / math.sqrt(self.features_out), 1. / math.sqrt(self.features_out)
        self.weight = tf.Variable(tf.random.uniform([self.features_in, self.features_out], minval=-stdv_weight, maxval=stdv_weight, dtype=self.dtype), name='w')
        self.bias = tf.Variable(tf.random.uniform([self.features_out], minval=-stdv_bias, maxval=stdv_bias, dtype=self.dtype), name='b')

    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.features_out), 1. / math.sqrt(self.features_out)
        self.weight.assign(tf.random.uniform([self.features_in, self.features_out], minval=-stdv_weight, maxval=stdv_weight, dtype=self.dtype))
        self.bias.assign(tf.random.uniform([self.features_out], minval=-stdv_bias, maxval=stdv_bias, dtype=self.dtype))

    @tf.function
    def __call__(self, input, adj, eye=False):
        XW = tf.matmul(input, self.weight)
        if eye:
            return XW + self.bias
        elif type(adj) == tf.SparseTensor:
            return tf.sparse.sparse_dense_matmul(adj, XW) + self.bias
        else:
            return tf.matmul(adj, XW) + self.bias

class LAYER_TRUNCATED_KRYLOV(LAYER_GENERAL):
    def __init__(self, features_in, features_out, n_blocks, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None, name=None, dtype=tf.float32):
        super(LAYER_TRUNCATED_KRYLOV, self).__init__(name=name)
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.features_in, self.features_out, self.n_blocks = features_in, features_out, n_blocks
        self.dtype = dtype
        stdv_weight, stdv_bias = 1. / math.sqrt(self.features_out), 1. / math.sqrt(self.features_out)
        self.weight = tf.Variable(tf.random.uniform([self.features_in * self.n_blocks, self.features_out], minval=-stdv_weight, maxval=stdv_weight, dtype=self.dtype), name='w')
        self.bias = tf.Variable(tf.random.uniform([self.features_out], minval=-stdv_bias, maxval=stdv_bias, dtype=self.dtype), name='b')

    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.features_out), 1. / math.sqrt(self.features_out)
        self.weight.assign(tf.random.uniform([self.features_in * self.n_blocks, self.features_out], minval=-stdv_weight, maxval=stdv_weight, dtype=self.dtype))
        self.bias.assign(tf.random.uniform([self.features_out], minval=-stdv_bias, maxval=stdv_bias, dtype=self.dtype))

    @tf.function
    def __call__(self, input, adj, eye=True):
        if self.n_blocks == 1:
            output = tf.matmul(input, self.weight)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = tf.matmul(self.LIST_A_EXP_X_CAT, self.weight)
        elif self.LIST_A_EXP is not None: # TODO: TK more than 1 layer not tested!
            feature_output = []
            for i in range(self.n_blocks):
                AX = tf.sparse.sparse_dense_matmul(self.LIST_A_EXP[i], input)
                feature_output.append(AX)
            output = tf.matmul(tf.concat(feature_output, 1), self.weight)
        if eye:
            return output + self.bias
        elif type(adj) == tf.SparseTensor:
            return tf.sparse.sparse_dense_matmul(adj, output) + self.bias
        else:
            return tf.matmul(adj, output) + self.bias