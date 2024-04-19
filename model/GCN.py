import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, adjs, dim_in, dim_out):
        super(GCN, self).__init__()
        self.adj = adjs
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.view_num = len(adjs)
        # if dim_in > 512:
        #     hidden_layer = 128
        # else:
        #     hidden_layer = 32
        hidden_layer = dim_in // 2
        self.W1 = nn.Linear(dim_in, hidden_layer, bias=False)  # set half hidden weight
        self.W2 = nn.Linear(hidden_layer, dim_out, bias=False)
        self.W = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, X):
        Z = F.relu(self.W1(self.adj.mm(X)))  # relu
        Z = F.dropout(Z, 0.3)
        Z = self.W2(self.adj.mm(Z))
        return Z


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class JdcGCN(nn.Module):
    def __init__(self, adj_list, com_adj, view, dim_out):
        super(JdcGCN, self).__init__()
        self.view = view
        self.MyGCN = []
        sum_view = 0
        for v in range(len(self.view)):
            sum_view += view[v]
            setattr(self, 'gcnv{}'.format(v), GCN(adj_list[v], view[v], dim_out))
        self.ComGCN = GCN(com_adj, sum_view, dim_out)
        self.attention = Attention(dim_out)

    def forward(self, features, com_fea):
        emb = []
        for v in range(len(self.view)):
            attr = getattr(self, 'gcnv{}'.format(v))
            emb.append(attr(features[v]))
        emb.append(self.ComGCN(com_fea))
        emb, att = self.attention(torch.stack(emb, dim=1))
        return emb, att


class ClusteringJdcGCN(nn.Module):
    def __init__(self, adj_list, com_adj, view, dim_out, n_clusters, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2,
                 n_dec_3):
        super(ClusteringJdcGCN, self).__init__()
        self.view = view
        self.MyGCN = []
        self.alpha = 1.0
        # self.actfun = nn.ReLU()
        sum_view = 0
        for v in range(len(self.view)):
            sum_view += view[v]
            setattr(self, 'gcnv{}'.format(v), GCN(adj_list[v], view[v], dim_out))
        self.ComGCN = GCN(com_adj, sum_view, dim_out)
        self.attention = Attention(dim_out)
        # cluster layer
        # self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, dim_out))
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=dim_out,
            n_z=n_clusters)

    def forward(self, features, com_fea):
        emb_list = []
        for v in range(len(self.view)):
            attr = getattr(self, 'gcnv{}'.format(v))
            emb_list.append(attr(features[v]))
        emb_list.append(self.ComGCN(com_fea))
        emb, att = self.attention(torch.stack(emb_list, dim=1))
        rec_emb, z = self.ae(emb)

        return emb_list, emb, rec_emb, z, att


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z
