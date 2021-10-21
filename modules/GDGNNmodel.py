import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .encoder import *
from torch_scatter import scatter
from torch_sparse import SparseTensor
import ipdb

def Expectation_log_Dirichlet(gamma):
    #E_{Dir(\theta| \gamma)}[\log \theta] = \Psi(\gamma) - \Psi(|\gamma|)
    return torch.digamma(gamma) -torch.digamma(gamma.sum(dim=-1, keepdim=True))

class GDGNNModel(nn.Module):

    def __init__(self, args, word_vec, whole_edge):
        super(GDGNNModel, self).__init__()
        # encoder
        self.args = args
        if word_vec is None:
            self.word_vec = nn.Parameter(torch.Tensor(args.vocab_size, args.ni))
            nn.init.normal_(self.word_vec, std=0.01)
        else:
            self.word_vec = nn.Parameter(torch.tensor(word_vec))
            if args.fixing:
                self.word_vec.requires_grad = False

        if args.prior_type == 'Dir2':
            self.encoder = GNNDir2encoder(args, self.word_vec)
        elif args.prior_type == 'Gaussian':
            self.encoder = GNNGaussianencoder(args, self.word_vec)
        else:
            raise ValueError("the specific prior type is not supported")

        self.word_vec_beta = nn.Parameter(torch.Tensor(args.vocab_size, args.ni))
        nn.init.normal_(self.word_vec_beta, std=0.01)
        self.topic_vec = nn.Parameter(torch.Tensor(args.num_topic, args.ni))
        nn.init.normal_(self.topic_vec)

        self.topic_edge_vec = nn.Parameter(torch.Tensor(args.num_topic, 2 * args.ni))
        self.noedge_vec = nn.Parameter(torch.Tensor(1, 2 * args.ni))
        nn.init.normal_(self.topic_edge_vec)
        nn.init.normal_(self.noedge_vec, std=0.01)

        edge_size = whole_edge.size(1)
        self.whole_edge = whole_edge

        self.maskrate = torch.tensor(1.0 / (args.num_topic), dtype=torch.float, device=args.device)
        # self.maskrate = torch.tensor(0.5, dtype=torch.float, device=args.device)

        self.topic_linear = nn.Linear(3 * args.ni, 64,bias=False)

        self.enc_params= list(self.encoder.parameters())
        self.dec_params = [self.word_vec_beta, self.topic_vec,self.topic_edge_vec,self.noedge_vec] + list(self.topic_linear.parameters())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.topic_linear.weight, std=0.01)
        # nn.init.constant_(self.topic_linear.weight, val=0)
        # nn.init.constant_(self.topic_linear.bias, val=0)
        pass

    def forward(self, batch_data):
        idx_x = batch_data.x
        x_batch = batch_data.x_batch
        idx_w = batch_data.x_w
        edge_w = batch_data.edge_w
        edge_id = batch_data.edge_id
        edge_id_batch = batch_data.edge_id_batch
        edge_index = batch_data.edge_index 
        size = int(x_batch.max().item() + 1)

        # compute posterior
        #####KL theta
        param, phi = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w)  # (B, K ) (B*len, K) B*len = B*len(idx_x)

        KL1 = self.encoder.KL_loss(param)

        #####KL(z)
        # (batch, max_length)
        if self.args.prior_type in ['Dir2', 'Gaussian']:
            theta = self.encoder.reparameterize(param)
            KL2 = torch.sum(phi * ((phi / (theta[x_batch] + 1e-10) + 1e-10).log()), dim=-1)  # (B*len)
        if self.args.prior_type == 'Dir':
            gamma = param[0] # (B, K)
            Elogtheta = Expectation_log_Dirichlet(gamma) # (B, K)
            KL2 = torch.sum(phi * ((phi+1e-10).log() - Elogtheta[x_batch]), dim=-1)
        KL2 = myscattersum(idx_w * KL2, x_batch, size=size)  # B

        if not self.args.eval and self.args.prior_type in ['Dir2','Gaussian'] and self.args.iter_ < self.args.iter_threahold:
            phi = theta[x_batch]
        


        ##### generate structure
        W = self.get_W()  # (K, K)  after sigmod
        log_PW = (W).log()  # K,K
        log_NW = (1 - W).log()  # K,K
        p_phi = phi[edge_index, :]  # 2,B*len_edge, K
        p_edge = torch.sum(torch.matmul(p_phi[0], log_PW) * p_phi[1], dim=-1)  # B*len_edge
        p_edge = myscattersum(p_edge, edge_id_batch, size=size)  # B

        

        neg_mask,neg_mask2 = adj_mask(x_batch, device=idx_w.device)
        
        neg_mask[edge_index[0], edge_index[1]] = 0

        n_edge = torch.matmul(torch.matmul(phi, log_NW), phi.T)  # B*len, B*len
        n_edge1 = torch.sum(n_edge * neg_mask, dim=-1)  # B*len
        n_edge1 = myscattersum(n_edge1, x_batch, size=size)  # B

        tmp = torch.ones_like(edge_id_batch, dtype=torch.float, device=edge_id_batch.device)
        NP = myscattersum(tmp, edge_id_batch, size=size) #B

        NN = myscattersum(torch.sum(neg_mask, dim=-1), x_batch,size=size)#B

        recon_structure = -(p_edge + n_edge1 / (NN + 1e-6) * NP )


        # #### recon_word
        beta = self.get_beta()  # (K, V)

        q_z = torch.distributions.RelaxedOneHotCategorical(temperature=self.args.temp, probs=phi)
        z = q_z.rsample([self.args.num_samp])  # (ns, B*len, K)

        z = hard_z(z, dim=-1)

        beta_s = beta[:, idx_x]  # K*(B*len) !TODO idx_x or idx_x
        beta_s = beta_s.permute(1, 0)  # (B*len)*K
        recon_word = (phi * (beta_s + 1e-6).log()).sum(-1)  # (B*len)
        recon_word = -myscattersum(idx_w * recon_word, x_batch)  # B

        ######recon_edge
        beta_edge = self.get_beta_edge(weight=False)

        edge_w = torch.unsqueeze(edge_w, 0)  # (1, B*len_edge)
        beta_edge_s = (beta_edge[:, edge_id]).permute(1, 0)  # (B*len_edge,K)
        

        z_edge_w = z[:, edge_index, :]  # (ns,2,B*len_edge, K)
        mask = z_edge_w > self.maskrate
        z_edge_w = z_edge_w * mask

        edge_w = edge_w.unsqueeze(-1)  # (1, B*len_edge, 1)
        z_edge_w = edge_w * z_edge_w.prod(dim=1)  # (ns,B*len_edge, K)
        beta_edge_s = (beta_edge_s.unsqueeze(0) * z_edge_w)  # (ns,B*len_edge,K)

        beta_edge_s = beta_edge_s.permute(1, 0, 2)  # (B*len_edge, ns,K)
        # beta_edge_s = myscattersum(beta_edge_s, edge_id_batch,size=size)  # (B, ns,K)
        beta_edge_s = scatter(beta_edge_s, edge_id_batch, dim=0, dim_size=size, reduce='sum')
        beta_edge_s = beta_edge_s.permute(1, 0, 2)  # (ns,B,K)

        z_edge_w = z_edge_w.permute(1, 0, 2)  # (B*len,ns,K)
        # z_edge_w = myscattersum(z_edge_w, edge_id_batch,size=size)  # (B,ns,K)
        z_edge_w = scatter(z_edge_w, edge_id_batch, dim=0, dim_size=size, reduce='sum')
        z_edge_w = z_edge_w.permute(1, 0, 2)  # (ns,B,K)
        recon_edge = -(torch.clamp_min(beta_edge_s, 1e-10) / torch.clamp_min(z_edge_w, 1e-10)).log().sum(-1).mean(0)

        if not self.args.eval and self.args.prior_type in ['Dir2', 'Gaussian'] and self.args.iter_ < self.args.iter_threahold:

            loss = recon_word + KL1
        else:
            loss = (recon_edge + recon_word + KL1 + KL2 +recon_structure)
        

        if torch.isnan(loss).sum() > 0 or loss.mean() > 1e20 or torch.isnan(recon_structure).sum() > 0:
            ipdb.set_trace()

        outputs = {
            "loss": loss.mean(),
            "recon_word": recon_word.mean(),
            "recon_edgew": recon_edge.mean(),
            "recon_structure": recon_structure.mean(),
            "p_edge": p_edge.mean(),
            "KL1": KL1.mean(),
            "KL2": KL2.mean()
        }

        return outputs

    def loss(self, batch_data):
        return self.forward(batch_data)

    def get_beta(self):
        beta = torch.matmul(self.topic_vec, self.word_vec_beta.T)
        beta = torch.softmax(beta, dim=-1)
        return beta

    def get_W(self):
        tew_vector = torch.cat([self.topic_vec, self.topic_edge_vec], dim=-1)
        topic_vec = self.topic_linear(tew_vector)  # (K, 64)
        head_vec, tail_vec = torch.chunk(topic_vec, 2, dim=-1)  # (K, nw) (K,nw)
        head_vec = L2_norm(head_vec)
        tail_vec = L2_norm(tail_vec)
        W = torch.matmul(head_vec, tail_vec.T)  # (K, K)
        W = torch.sigmoid(4*W)
        I = torch.eye(self.args.num_topic, dtype=torch.float, device=self.args.device)
        mask = 1-I
        return torch.clamp(W, 1e-4, 1-1e-4)

    def get_beta_edge(self, weight=True):
        beta = self.get_beta()
        beta_edge_w = beta[:, self.whole_edge]  # (K,2,edge_size)
        beta_edge_w = beta_edge_w.prod(dim=1)  # (K,edge_size)
        beta_nedge_w = 1 - beta_edge_w.sum(-1)  # K
        beta_edge_w = torch.cat([beta_nedge_w.unsqueeze(1), beta_edge_w], dim=1)  # (K,1+edge_size) 0处存放非边的权重
        edge_vec = self.word_vec_beta[self.whole_edge, :]  # (2, edge_size , nw)
        edge_vec = torch.cat([edge_vec[0], edge_vec[1]], dim=-1)  # (edge_size ,2*nw)
        edge_vec = torch.cat([self.noedge_vec, edge_vec], dim=0)  # (edge_size+1 ,2*nw)
        beta_edge = torch.matmul(self.topic_edge_vec, edge_vec.T)  # (K, edge_size+1)
        beta_edge = weightedSoftmax(beta_edge, beta_edge_w)
        if weight:
            beta_edge = beta_edge * beta_edge_w  ## (K,edge_size+1)
        return beta_edge

    def get_degree(self, weight=True):
        beta_edge = self.get_beta_edge(weight).permute(1, 0)  ##(1+edge_size,K)
        beta_matrix = SparseTensor(row=self.whole_edge[0], col=self.whole_edge[1],
                                   value=beta_edge[1:, :],
                                   sparse_sizes=(self.args.vocab_size + 1, self.args.vocab_size + 1))
        out_degree = beta_matrix.sum(0)  # vocab_size *K
        in_degree = beta_matrix.sum(1)  # vocab_size *K
        degree = out_degree + in_degree  # vocab_size *K
        degree = degree.permute(1, 0)  # K * vocab_size

        return degree

    def get_doctopic(self, batch_data):
        idx_x = batch_data.x
        x_batch = batch_data.x_batch
        idx_w = batch_data.x_w
        edge_w = batch_data.edge_w
        edge_index = batch_data.edge_index  # edge_index 中的idx与idx_x对应上
        param, phi = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w)  # (B, K ) (B*len, K) B*len = B*len(idx_x)
        Ns = myscattersum(torch.ones_like(idx_w,device=idx_w.device, dtype=torch.float), x_batch) #B
        phis = myscattersum(phi*torch.unsqueeze(idx_w,dim=-1),x_batch) #B*K
        theta =phis/torch.unsqueeze(Ns, dim=-1)
        # theta = self.encoder.reparameterize(param)
        return theta  # (B, K )

    def tdregular(self):
        beta = self.get_beta()
        topic_vec = L2_norm(beta)
        cos = torch.sum(topic_vec.unsqueeze(0)*topic_vec.unsqueeze(1), dim =-1) #K*K
        mean = torch.asin(torch.sqrt(torch.det(cos)))
        var = (math.pi/2-torch.sqrt(torch.det(cos)))**2
        return mean-var

def L2_norm(vec):
    #vec M*V
    nvec = torch.sqrt(torch.sum(vec**2, dim=-1, keepdim=True))
    return vec/torch.clamp_min(nvec, 1e-10)



def weightedSoftmax(x, w):
    # x: B*L w: B*L
    # x = x - torch.unsqueeze(x.min(-1)[0], -1)
    logwsum = torch.logsumexp(x + (w + 1e-20).log(), dim=-1, keepdim=True)
    if torch.isnan(logwsum).sum() > 0:
        import ipdb
        ipdb.set_trace()

    return (x - logwsum).exp()


def to_BOW(idx, idx_w, idx_batch, V):
    device = idx.get_device()
    if device >= 0:
        embeddings = torch.eye(V, device=device)
    else:
        embeddings = torch.eye(V)
    batch_data = embeddings[idx]
    batch_data = torch.unsqueeze(idx_w, 1) * batch_data
    size = int(idx_batch.max().item() + 1)
    batch_data = scatter(batch_data, index=idx_batch, dim=-2, dim_size=size, reduce='sum')
    return batch_data

def hard_z(y_soft, dim):
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


def myscattersum(src, index, size=None):
    # src: B*len, C1, C2,... index: B*len
    # return: B, C1,C2
    if src is None:
        src = torch.ones_like(index, dtype=torch.float, device=index.device)
    device = src.device
    flag = (index[1:] - index[:-1]).long()  # BLEN-1
    flag = torch.cat([flag, torch.ones(1, dtype=torch.long, device=device)], dim=0)  # LEN
    RANGE = torch.ones_like(index, dtype=torch.long)  # BLEN
    RANGE = RANGE.cumsum(dim=0) - 1  # BELN

    ids = torch.where(flag > 0)[0]  # LEN #每个样本序列最后一个点的坐标
    nids = ids[1:] - ids[:-1]  # LEN-1
    nids = torch.cat([ids[[0]] + 1, nids], dim=0)  # LEN  每个样本序列有多少个点
    ids = ids[:-1] + 1  # LEN-1

    yindex = torch.zeros_like(index, dtype=torch.long)
    yindex[ids] = nids[:-1]
    yindex = yindex.cumsum(dim=0)
    yindex = RANGE - yindex

    indexs = torch.stack([index, yindex], dim=0)
    ST = torch.sparse_coo_tensor(indexs, src,device=device)
    ret = ST.to_dense().sum(dim=1)  # B,C1,C2
    if size is not None and size >ret.size(0):
        N = size - ret.size(0)
        zerovec=torch.zeros(N,*ret.size()[1:], dtype=torch.float,device=device)
        ret = torch.cat([ret, zerovec], dim = 0)

    return ret


def adj_mask(x_batch, device):
    size = torch.max(x_batch)
    N = x_batch.size(0)
    # mask = torch.ones((N, N), device=device)
    # mask = torch.dropout(mask,p=0.5, train=True)
    mask = torch.zeros((N, N), device=device)
    for i in range(size + 1):
        idxs = torch.where(x_batch == i)[0]
        mask[idxs[0]:idxs[-1] + 1, idxs] = 1
    mask2 = 1-mask
    diag = torch.diag(mask)
    a_diag = torch.diag_embed(diag)
    mask = mask-a_diag
    return mask, mask2


if __name__ == '__main__':
    import torch

    x = torch.rand(14, 3)
    # x_batch = torch.zeros(10)
    # ids = torch.randint(0, 10, [100])
    # x_batch[ids] = 1
    # x_batch = x_batch.cumsum(dim=-1).long()
    x_batch = torch.tensor([0,0,0,1,1,1,2,2,4,4,4,5,5,5])
    ret = myscattersum(x, x_batch)
    import ipdb
    ipdb.set_trace()
