import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import general_GCN_layer, snowball_layer, truncated_krylov_layer
import numpy as np

class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()

# class linear_snowball(graph_convolutional_network):
#     def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
#         super(linear_snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
#         for k in range(nlayers):
#             self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
#         self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
#     def forward(self, x, adj):
#         list_output_blocks = []
#         for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
#             if layer_num == 0:
#                 list_output_blocks.append(F.dropout(layer(x, adj), self.dropout, training=self.training))
#             else:
#                 list_output_blocks.append(F.dropout(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj), self.dropout, training=self.training))
#         output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
#         return F.log_softmax(output, dim=1)

class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)), self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        return F.log_softmax(output, dim=1)

class truncated_krylov(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        LIST_A_EXP, LIST_A_EXP_X, A_EXP = [], [], torch.eye(adj.size()[0], dtype=adj.dtype).cuda()
        if str(adj.layout) == 'torch.sparse_coo':
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj
        for _ in range(n_blocks):
            if nlayers > 1:
                indices = torch.nonzero(A_EXP).t()
                values = A_EXP[indices[0], indices[1]]
                LIST_A_EXP.append(torch.sparse.FloatTensor(indices, values, A_EXP.size()))
            LIST_A_EXP_X.append(torch.mm(A_EXP, features))
            torch.cuda.empty_cache()
            A_EXP = torch.mm(A_EXP, dense_adj)
        self.hidden.append(truncated_krylov_layer(nfeat, n_blocks, nhid, LIST_A_EXP_X_CAT=torch.cat(LIST_A_EXP_X, 1)))
        for _ in range(nlayers - 1):
            self.hidden.append(truncated_krylov_layer(nhid, n_blocks, nhid, LIST_A_EXP=LIST_A_EXP))
        self.out = truncated_krylov_layer(nhid, 1, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(list_output_blocks[layer_num - 1], adj)), self.dropout, training=self.training))
        output = self.out(list_output_blocks[self.nlayers - 1], adj, eye=True)
        return F.log_softmax(output, dim=1)