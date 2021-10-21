import math
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GatedGraphConv, GraphConv, GCNConv
from torch.autograd import Variable
from torch_sparse import SparseTensor
import numpy as np

class GNNDir2encoder(nn.Module):

    def __init__(self, args, word_vec):
        super(GNNDir2encoder, self).__init__()
        self.args= args
        if word_vec is not None and args.word:
            self.word_vec = word_vec
            if args.fixing:
                self.word_vec.requires_grad = False
        else:
            # self.word_vec = nn.Parameter(torch.Tensor(args.vocab, args.nw))
            self.word_vec = torch.eye(args.vocab_size, dtype=torch.float, device=args.device)
        input_size = self.word_vec.size(1)
        # self.enc1_gnn = GatedGraphConv(args.nw, num_layers=2, bias=True)  # 1995 -> 100
        self.enc1_gnn1 = GraphConv(input_size,args.nw,  bias=True)
        self.bn_gnn1 = nn.BatchNorm1d(args.nw)
        # self.enc1_gnn2 = GraphConv(args.nw,args.nw,  bias=True)
        # self.bn_gnn2 = nn.BatchNorm1d(args.nw)

        self.enc2_fc1 = nn.Linear(input_size+args.nw, args.enc_nh)
        self.enc2_fc2 = nn.Linear(input_size+args.nw, args.enc_nh)
        self.enc2_drop = nn.Dropout(0.2)

        self.mean_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(args.num_topic)  # bn for mean
        self.logvar_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)  # bn for logvar

        self.phi_fc = nn.Linear(args.nw+ input_size + args.enc_nh, args.num_topic)
        self.phi_bn = nn.BatchNorm1d(args.num_topic)

        self.logvar_bn.weight.requires_grad = False
        nn.init.constant_(self.logvar_bn.weight, 1.0)
        self.mean_bn.weight.requires_grad = False
        nn.init.constant_(self.mean_bn.weight, 1.0)

        prior_mean = torch.Tensor(1, args.num_topic).fill_(0)
        prior_var = torch.Tensor(1, args.num_topic).fill_(args.variance)
        # self.a = args.prior * np.ones((1, args.num_topic)).astype(np.float32)
        # prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        # prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T +
        #                               (1.0 / (args.num_topic * args.num_topic) * np.sum(1.0 / self.a, 1)).T))
        # prior_mean = torch.Tensor(1, args.num_topic).fill_(0)
        #
        # prior_var = torch.Tensor(1, args.num_topic).fill_(1.0 / self.args.prior * (1 - 1.0/args.num_topic))
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # self.propa = GCNConv(args.num_topic, args.num_topic, bias=False)
        # nn.init.eye_(self.propa.weight)
        # self.propa.weight.requires_grad=False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.logvar_fc.weight)
        nn.init.zeros_(self.logvar_fc.bias)
        pass

    def forward(self, idx_x, idx_w, x_batch, edge_index, edge_weight):
        x = self.word_vec[idx_x]  # N*nw
        diag = torch.ones(2, idx_x.size(0), dtype=torch.long, device=idx_x.device).cumsum(dim=-1) - 1
        edge_index_exp = torch.cat([edge_index, diag], dim=-1)  # 添加对角线
        diag_w = torch.ones(idx_x.size(0), dtype=torch.float, device=idx_x.device) * idx_w
        edge_weight_exp = torch.cat([edge_weight, diag_w], dim=0)

        enc1 = torch.tanh(self.bn_gnn1(self.enc1_gnn1(x, edge_index_exp, edge_weight=edge_weight_exp)))


        if torch.isnan(enc1).sum()>0:
            import ipdb
            ipdb.set_trace()
        enc1 = torch.cat([enc1, x], dim=-1)
        enc2 = torch.sigmoid(self.enc2_fc1(enc1)) * torch.tanh(self.enc2_fc2(enc1))
        size = int(x_batch.max().item() + 1)
        enc2 = scatter(enc2, x_batch, dim=0, dim_size=size, reduce='sum')  # B*enc_h
        enc2d = self.enc2_drop(enc2)


        mean = self.mean_bn(self.mean_fc(enc2d))  # posterior mean
        logvar = self.logvar_fc(enc2d) # posterior log variance

        word_embed = torch.cat([enc1, enc2[x_batch]], dim=-1)
        phi = torch.softmax(self.phi_fc(word_embed), dim=-1)  # (B*max_len)*num_topic


        param = (mean,logvar)
        if torch.isnan(mean).sum()>0:
            import ipdb
            ipdb.set_trace()
        return param, phi

    def reparameterize(self, param):
        posterior_mean = param[0]
        posterior_var = param[1].exp()
        # take sample
        if self.training:
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        else:
            z = posterior_mean
        theta = torch.softmax(z, dim=-1)
        return theta

    def KL_loss(self, param):
        posterior_mean = param[0]
        posterior_logvar = param[1]

        prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KL = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.args.num_topic)

        if torch.isinf(KL).sum()>0 or torch.isnan(KL).sum()>0:
            import ipdb
            ipdb.set_trace()

        return KL


class GNNGaussianencoder(nn.Module):

    def __init__(self, args, word_vec):
        super(GNNGaussianencoder, self).__init__()
        self.args= args
        if word_vec is not None:
            self.word_vec = word_vec
            if args.fixing:
                self.word_vec.requires_grad = False
        else:
            self.word_vec = nn.Parameter(torch.Tensor(args.vocab, args.nw))
        input_size = self.word_vec.size(1)
        self.enc1_gnn1 = GraphConv(input_size, args.nw, bias=True)
        self.bn_gnn1 = nn.BatchNorm1d(args.nw)
        # self.enc1_gnn2 = GraphConv(args.nw,args.nw,  bias=True)
        # self.bn_gnn2 = nn.BatchNorm1d(args.nw)

        self.enc2_fc1 = nn.Linear(input_size + args.nw, args.enc_nh)
        self.enc2_fc2 = nn.Linear(input_size + args.nw, args.enc_nh)
        self.enc2_drop = nn.Dropout(0.2)

        self.mean_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(args.num_topic)  # bn for mean
        self.logvar_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50

        self.phi_fc = nn.Linear(args.nw + input_size + args.enc_nh, args.num_topic)
        self.phi_bn = nn.BatchNorm1d(args.num_topic)

        self.decoder = nn.Linear(args.num_topic, args.num_topic)  # 50   -> 1995


        if args.init_mult != 0:
            # std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, args.init_mult)
        # self.mean_bn.weight.requires_grad = False
        # nn.init.constant_(self.mean_bn.weight, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.logvar_fc.weight)
        nn.init.zeros_(self.logvar_fc.bias)
        pass

    def forward(self, idx_x, idx_w, x_batch, edge_index, edge_weight):
        x = self.word_vec[idx_x]  # N*nw
        diag = torch.ones(2, idx_x.size(0), dtype=torch.long, device=idx_x.device).cumsum(dim=-1) - 1
        edge_index_exp = torch.cat([edge_index, diag], dim=-1)  # 添加对角线
        diag_w = torch.ones(idx_x.size(0), dtype=torch.float, device=idx_x.device) * idx_w
        edge_weight_exp = torch.cat([edge_weight, diag_w], dim=0)

        # enc1 = torch.tanh(self.enc1_gnn1(x, edge_index_exp, edge_weight=edge_weight_exp))  # N
        enc1 = torch.tanh(self.bn_gnn1(self.enc1_gnn1(x, edge_index_exp, edge_weight=edge_weight_exp)))
        # enc1 = torch.tanh(self.bn_gnn2(self.enc1_gnn2(enc1+x, edge_index_exp, edge_weight=edge_weight_exp)))
        # enc1 = self.enc1_gnn2(enc1+x, edge_index, edge_weight=edge_weight)

        if torch.isnan(enc1).sum() > 0:
            import ipdb
            ipdb.set_trace()
        enc1 = torch.cat([enc1, x], dim=-1)
        enc2 = torch.sigmoid(self.enc2_fc1(enc1)) * torch.tanh(self.enc2_fc2(enc1))
        size = int(x_batch.max().item() + 1)
        enc2 = scatter(enc2, x_batch, dim=0, dim_size=size, reduce='sum')  # B*enc_h
        # enc2 = myscattersum(enc2, x_batch)
        enc2d = self.enc2_drop(enc2)

        # mean = self.mean_bn(self.mean_fc(enc2d))  # posterior mean
        mean = self.mean_fc(enc2)  # posterior mean
        logvar = self.logvar_fc(enc2d)  # posterior log variance
        # logvar = self.logvar_bn(self.logvar_fc(enc2)) # posterior log variance

        # word_embed = torch.cat([torch.unsqueeze(idx_w, 1)*x, enc2[x_batch]], dim=-1)
        word_embed = torch.cat([enc1, enc2[x_batch]], dim=-1)
        # phi = torch.softmax(self.phi_bn(self.phi_fc(word_embed)), dim=-1)  # (B*max_len)*num_topic
        phi = torch.softmax(self.phi_fc(word_embed), dim=-1)  # (B*max_len)*num_topic
        # phi = self.propa(phi, edge_index, edge_weight)

        # logvar = torch.clamp(logvar, -10, 10)
        param = (mean, logvar)
        if torch.isnan(mean).sum() > 0:
            import ipdb
            ipdb.set_trace()
        return param, phi

    def reparameterize(self, param):
        posterior_mean = param[0]
        posterior_var = param[1].exp()
        # take sample
        if self.training:
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        else:
            z = posterior_mean
        theta = torch.softmax(self.decoder(z), dim=-1)
        return theta

    def KL_loss(self, param):
        posterior_mean = param[0]
        posterior_logvar = param[1]
        KL = -0.5 * torch.sum(1 - posterior_mean ** 2 + posterior_logvar -
                              torch.exp(posterior_logvar), dim=1)
        return KL




