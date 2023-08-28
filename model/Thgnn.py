from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math

class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        masked_weight = torch.mul(weight, adj_mat).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GraphAttnSemIndividual(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, requires_weight=False):
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        if requires_weight:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=6, out_features=8, num_heads=8, hidden_dim=64, num_layers=1):
        super(StockHeteGAT, self).__init__()
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=False):
        _, support = self.encoding(inputs)
        support = support.squeeze()
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        all_embedding = self.pn(all_embedding)
        if requires_weight:
            return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
        else:
            return self.predictor(all_embedding)
