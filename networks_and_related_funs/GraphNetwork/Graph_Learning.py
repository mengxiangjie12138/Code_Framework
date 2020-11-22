import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphLearning(Module):
    def __init__(self, in_features):
        super(GraphLearning, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        s = torch.zeros((len(inputs), len(inputs)))
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                s[i, j] = torch.exp(F.relu(self.weight.mm(torch.abs(inputs[i] - inputs[j]).unsqueeze(0).t())))[0][0]
        A = F.softmax(s, dim=1)
        D = torch.diag(torch.sum(A, dim=1))
        return A, D

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.in_features) + ')'
