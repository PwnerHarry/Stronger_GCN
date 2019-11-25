import numpy as np
import scipy.sparse as sp
import random
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, math
import tensorflow as tf

def sparse_matrix2tensor(X, dtype):
    if dtype == tf.float32:
        dtype = np.float32
    elif dtype == tf.float16:
        dtype = np.float16
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, tf.constant(coo.data.astype(dtype)), coo.shape)

def dense2sparse(X):
    idx = tf.where(tf.not_equal(X, 0))
    return tf.SparseTensor(idx, tf.gather_nd(X, idx), X.get_shape())

def split_dataset(num_nodes, labels, dataset, public, percent):
    labels_np = labels.numpy()
    if public == 1:
        if dataset == 'cora':
            idx_train, idx_val, idx_test = np.arange(140).reshape(-1, 1), np.arange(140, 640).reshape(-1, 1), np.arange(1708, 2708).reshape(-1, 1)
        elif dataset == 'citeseer':
            idx_train, idx_val, idx_test = np.arange(120).reshape(-1, 1), np.arange(120, 620).reshape(-1, 1), np.arange(2312, 3312).reshape(-1, 1)
        elif dataset == 'pubmed':
            idx_train, idx_val, idx_test = np.arange(60).reshape(-1, 1), np.arange(60, 560).reshape(-1, 1), np.arange(18717, 19717).reshape(-1, 1)
    elif public == 2:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels_np)
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train,random.sample(list(np.where(labels_np==c)[0].astype(int)), 20)])
        others = (np.delete(all_data.astype(int), idx_train.astype(int)))
        for c in all_class:
            idx_val = np.hstack([idx_val,random.sample(list(np.where(labels_np[others]==c)[0].astype(int)), math.ceil(500/all_class.shape[0]) )])
        others = (np.delete(others.astype(int), idx_val.astype(int)))
        for c in all_class:
            idx_test = np.hstack([idx_test,random.sample(list(np.where(labels_np[others]==c)[0].astype(int)), min(math.ceil(1000/all_class.shape[0]), np.where(labels_np[others]==c)[0].astype(int).shape[0]))])
    elif public == 0:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels_np)
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train, random.sample(list(np.where(labels_np==c)[0].astype(int)), math.ceil(np.where(labels_np==c)[0].shape[0]*percent))])
        others = (np.delete(all_data.astype(int), idx_train.astype(int)))
        for c in all_class:
            idx_val = np.hstack([idx_val, random.sample(list(np.where(labels_np[others]==c)[0].astype(int)), math.ceil(500/all_class.shape[0]) )])
        others = (np.delete(others.astype(int), idx_val.astype(int)))
        for c in all_class:
            idx_test = np.hstack([idx_test, random.sample(list(np.where(labels_np[others]==c)[0].astype(int)), min(math.ceil(1000/all_class.shape[0]), np.where(labels_np[others]==c)[0].astype(int).shape[0]))])
    idx_train, idx_val, idx_test = tf.convert_to_tensor(idx_train.astype(int), dtype=tf.int32), tf.convert_to_tensor(idx_val.astype(int), dtype=tf.int32), tf.convert_to_tensor(idx_test.astype(int), dtype=tf.int32)
    labels_train, labels_val, labels_test = tf.gather(labels, idx_train), tf.gather(labels, idx_val), tf.gather(labels, idx_test)
    return idx_train, idx_val, idx_test, labels_train, labels_val, labels_test

def load_dataset(dataset, dtype):
    adj, features, labels = sp.load_npz("%s_adj.npz" % dataset), np.load("%s_features.npy" % dataset), np.load("%s_labels.npy" % dataset)
    adj = sparse_matrix2tensor(adj, dtype=dtype)
    features = tf.constant(features, dtype=dtype)
    labels = tf.constant(labels, dtype=tf.int32)
    return adj, features, labels

def trainer():
    @tf.function
    def optimize_one_step(model, optimizer, features, adj, labels, idx_train, tracker_loss):
        with tf.GradientTape() as t:
            vars_trainable = model.vars_trainable()
            t.watch(vars_trainable)
            output = model(features, adj)
            labels_train, output_train = tf.gather(labels, idx_train), tf.gather(output, idx_train)
            loss_train = tracker_loss(labels_train, output_train)
            grads = t.gradient(loss_train, vars_trainable)
        optimizer.apply_gradients(zip(grads, vars_trainable))
        return loss_train
    return optimize_one_step

class DecoupledWeightDecayExtension(object):
    """This class allows to extend optimizers with decoupled weight decay.

    It implements the decoupled weight decay described by Loshchilov & Hutter
    (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
    decoupled from the optimization steps w.r.t. to the loss function.
    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.

    This class alone is not an optimizer but rather extends existing
    optimizers with decoupled weight decay. We explicitly define the two
    examples used in the above paper (SGDW and AdamW), but in general this
    can extend any OptimizerX by using
    `extend_with_decoupled_weight_decay(
        OptimizerX, weight_decay=weight_decay)`.
    In order for it to work, it must be the first class the Optimizer with
    weight decay inherits from, e.g.

    ```python
    class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
      def __init__(self, weight_decay, *args, **kwargs):
        super(AdamW, self).__init__(weight_decay, *args, **kwargs).
    ```

    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    def __init__(self, weight_decay, **kwargs):
        """Extension class that adds weight decay to an optimizer.

        Args:
            weight_decay: A `Tensor` or a floating point value, the factor by
                which a variable is decayed in the update step.
            **kwargs: Optional list or tuple or set of `Variable` objects to
                decay.
        """
        wd = kwargs.pop('weight_decay', weight_decay)
        super(DecoupledWeightDecayExtension, self).__init__(**kwargs)
        self._decay_var_list = None  # is set in minimize or apply_gradients
        self._set_hyper('weight_decay', wd)

    def get_config(self):
        config = super(DecoupledWeightDecayExtension, self).get_config()
        config.update({
            'weight_decay':
            self._serialize_hyperparameter('weight_decay'),
        })
        return config

    def minimize(self,
                 loss,
                 var_list,
                 grad_loss=None,
                 name=None,
                 decay_var_list=None):
        """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before
        applying then call `tf.GradientTape` and `apply_gradients()` explicitly
        instead of using this function.

        Args:
            loss: A callable taking no arguments which returns the value to
                minimize.
            var_list: list or tuple of `Variable` objects to update to
                minimize `loss`, or a callable returning the list or tuple of
                `Variable` objects. Use callable when the variable list would
                otherwise be incomplete before `minimize` since the variables
                are created at the first time `loss` is called.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            name: Optional name for the returned operation.
        Returns:
            An Operation that updates the variables in `var_list`.  If
            `global_step` was not `None`, that operation also increments
            `global_step`.
        Raises:
            ValueError: If some of the variables are not `Variable` objects.
        """
        self._decay_var_list = set(decay_var_list) if decay_var_list else False
        return super(DecoupledWeightDecayExtension, self).minimize(
            loss, var_list=var_list, grad_loss=grad_loss, name=name)

    def apply_gradients(self, grads_and_vars, name=None, decay_var_list=None):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            name: Optional name for the returned operation.  Default to the
                name passed to the `Optimizer` constructor.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
        Returns:
            An `Operation` that applies the specified gradients. If
            `global_step` was not None, that operation also increments
            `global_step`.
        Raises:
            TypeError: If `grads_and_vars` is malformed.
            ValueError: If none of the variables have gradients.
        """
        self._decay_var_list = set(decay_var_list) if decay_var_list else False
        return super(DecoupledWeightDecayExtension, self).apply_gradients(
            grads_and_vars, name=name)

    def _decay_weights_op(self, var):
        if not self._decay_var_list or var in self._decay_var_list:
            return var.assign_sub(
                self._get_hyper('weight_decay', var.dtype) * var,
                self._use_locking)
        return tf.no_op()

    def _decay_weights_sparse_op(self, var, indices):
        if not self._decay_var_list or var in self._decay_var_list:
            update = (-self._get_hyper('weight_decay', var.dtype) * tf.gather(
                var, indices))
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

    # Here, we overwrite the apply functions that the base optimizer calls.
    # super().apply_x resolves to the apply_x function of the BaseOptimizer.

    def _resource_apply_dense(self, grad, var):
        with tf.control_dependencies([self._decay_weights_op(var)]):
            return super(DecoupledWeightDecayExtension,
                         self)._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        decay_op = self._decay_weights_sparse_op(var, indices)
        with tf.control_dependencies([decay_op]):
            return super(DecoupledWeightDecayExtension,
                         self)._resource_apply_sparse(grad, var, indices)


def extend_with_decoupled_weight_decay(base_optimizer):
    """Factory function returning an optimizer class with decoupled weight
    decay.

    Returns an optimizer class. An instance of the returned class computes the
    update step of `base_optimizer` and additionally decays the weights.
    E.g., the class returned by
    `extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)` is
    equivalent to `tfa.optimizers.AdamW`.

    The API of the new optimizer class slightly differs from the API of the
    base optimizer:
    - The first argument to the constructor is the weight decay rate.
    - `minimize` and `apply_gradients` accept the optional keyword argument
      `decay_var_list`, which specifies the variables that should be decayed.
      If `None`, all variables that are optimized are decayed.

    Usage example:
    ```python
    # MyAdamW is a new class
    MyAdamW = extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
    # Create a MyAdamW object
    optimizer = MyAdamW(weight_decay=0.001, learning_rate=0.001)
    # update var1, var2 but only decay var1
    optimizer.minimize(loss, var_list=[var1, var2], decay_variables=[var1])

    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of 'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```

    Note: you might want to register your own custom optimizer using
    `tf.keras.utils.get_custom_objects()`.

    Args:
        base_optimizer: An optimizer class that inherits from
            tf.optimizers.Optimizer.

    Returns:
        A new optimizer class that inherits from DecoupledWeightDecayExtension
        and base_optimizer.
    """

    class OptimizerWithDecoupledWeightDecay(DecoupledWeightDecayExtension, base_optimizer):
        """Base_optimizer with decoupled weight decay.
        This class computes the update step of `base_optimizer` and
        additionally decays the variable with the weight decay being
        decoupled from the optimization steps w.r.t. to the loss
        function, as described by Loshchilov & Hutter
        (https://arxiv.org/pdf/1711.05101.pdf). For SGD variants, this
        simplifies hyperparameter search since it decouples the settings
        of weight decay and learning rate. For adaptive gradient
        algorithms, it regularizes variables with large gradients more
        than L2 regularization would, which was shown to yield better
        training loss and generalization error in the paper above.
        """

        def __init__(self, weight_decay, *args, **kwargs):
            # super delegation is necessary here
            super(OptimizerWithDecoupledWeightDecay, self).__init__(
                weight_decay, *args, **kwargs)

    return OptimizerWithDecoupledWeightDecay

class Adam(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
  def __init__(self, weight_decay, *args, **kwargs):
    super(Adam, self).__init__(weight_decay, *args, **kwargs)

class RMSprop(DecoupledWeightDecayExtension, tf.keras.optimizers.RMSprop):
  def __init__(self, weight_decay, *args, **kwargs):
    super(RMSprop, self).__init__(weight_decay, *args, **kwargs)
# @tf.function
# def accuracy(labels, output):
#     return tf.math.reduce_sum(tf.keras.metrics.sparse_categorical_accuracy(labels, output)) / len(labels)
