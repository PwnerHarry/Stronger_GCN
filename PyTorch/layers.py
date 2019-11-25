import math, torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)

class snowball_layer(general_GCN_layer):
    def __init__(self, in_features, out_features):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)
    
    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias
        else:
            return self.multiplication(adj, XW) + self.bias

class truncated_krylov_layer(general_GCN_layer):
    def __init__(self, in_features, n_blocks, out_features, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None):
        super(truncated_krylov_layer, self).__init__()
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight, self.output_bias = Parameter(torch.FloatTensor(self.in_features * self.n_blocks,self.out_features).cuda()), Parameter(torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(self.output_bias.size(0))
        torch.nn.init.uniform_(self.shared_weight, -stdv_shared_weight, stdv_shared_weight)
        torch.nn.init.uniform_(self.output_bias, -stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye=True):
        if self.n_blocks == 1:
            output = torch.mm(input, self.shared_weight)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = torch.mm(self.LIST_A_EXP_X_CAT, self.shared_weight)
        elif self.LIST_A_EXP is not None:
            feature_output = []
            for i in range(self.n_blocks):
                AX = self.multiplication(self.ADJ_EXPONENTIALS[i], input)
                feature_output.append(AX)
            output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias