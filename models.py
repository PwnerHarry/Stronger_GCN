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

class linear_snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(linear_snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.classifier = snowball_layer(nlayers * nhid + nfeat, nlayers * nhid + nfeat)
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(layer(x, adj), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj), self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye = 0)
        return F.log_softmax(output, dim = 1)

class linear_tanh_snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(linear_tanh_snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.classifier = snowball_layer(nlayers * nhid + nfeat, nlayers * nhid + nfeat)
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(layer(x, adj), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj), self.dropout, training=self.training))
        classifier = torch.tanh(self.classifier(torch.cat([x] + list_output_blocks, 1), adj, eye = 1))
        output = self.out(classifier, adj, eye = 0)
        return F.log_softmax(output, dim = 1)
    
    def reset_parameters(self):
        super(linear_tanh_snowball, self).reset_parameters()
        self.classifier.reset_parameters()

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
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye = 0)
        return F.log_softmax(output, dim = 1)

class truncated_krylov(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, ADJ_EXPONENTIALS):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        self.hidden.append(truncated_krylov_layer(nfeat, n_blocks, nhid, ADJ_EXPONENTIALS = ADJ_EXPONENTIALS))
        for _ in range(nlayers - 1):
            self.hidden.append(truncated_krylov_layer(nhid, n_blocks, nhid, ADJ_EXPONENTIALS = ADJ_EXPONENTIALS))
        self.classifier = truncated_krylov_layer(nhid, 1, nhid, ADJ_EXPONENTIALS = ADJ_EXPONENTIALS)
        self.out = truncated_krylov_layer(nhid,1,nclass, ADJ_EXPONENTIALS = ADJ_EXPONENTIALS)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(list_output_blocks[layer_num-1], adj)), self.dropout, training=self.training))
        classifier = torch.tanh(self.classifier(list_output_blocks[self.nlayers - 1], adj, eye = 1))
        output = self.out(classifier, adj, eye = 1)
        return F.log_softmax(output, dim = 1)
    
    def reset_parameters(self):
        super(truncated_krylov, self).reset_parameters()
        self.classifier.reset_parameters()
