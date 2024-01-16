import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn import TypedLinear
from openhgnn.utils import to_hetero_feat
from . import BaseModel, register_model


@register_model('SimpleHGN')
class SimpleHGN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.n_layers + [1]
        return cls(args.edge_dim,
                   len(hg.etypes),
                   [args.hidden_dim],
                   args.h_dim,
                   args.out_dim,
                   args.n_layers,
                   heads,
                   args.feats_drop_rate,
                   args.slope,
                   True,
                   args.beta
                   )

    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                 num_layers, heads, feat_drop, negative_slope,
                 residual, beta):
        super(SimpleHGN, self).__init__()
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim[0],
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, hg, h_dict, gamma):
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata='h')
            h = g.ndata['h']
            for l in range(self.num_layers):  # noqa E741
                h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True, gamma)
                h = h.flatten(1)

        h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        return h_dict


class SimpleHGNConv(nn.Module):

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False, gamma = None):
        emb = self.feat_drop(h)

        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1,
                                                                         self.num_heads, self.edge_dim)
        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if gamma is not None:
            edge_attention = edge_attention + gamma
        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            h_output = g.ndata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output