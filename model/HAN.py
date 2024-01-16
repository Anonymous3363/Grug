import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
# from . import GATConv
from . import BaseModel, register_model
from . import SemanticAttention
from . import MetapathConv
from openhgnn.utils.utils import extract_metapaths


@register_model('HAN')
class HAN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths_dict is None:
            meta_paths = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths = args.meta_paths_dict

        return cls(meta_paths=meta_paths, category=args.out_node_type,
                   in_size=args.hidden_dim, hidden_size=args.hidden_dim,
                   out_size=args.out_dim,
                   num_heads=args.num_heads,
                   dropout=args.dropout)

    def __init__(self, meta_paths, category, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.category = category
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.linear = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h_dict, gamma):
        for gnn in self.layers:
            h_dict = gnn(g, h_dict, gamma)
        out_dict = {ntype: self.linear(h_dict[ntype]) for ntype in self.category}
        return out_dict

        return {self.category: h.detach().cpu().numpy()}


class HANLayer(nn.Module):

    def __init__(self, meta_paths_dict, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_size, out_size, layer_num_heads,
                                          dropout, dropout, activation=F.elu,
                                          allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, gamma):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(
                    g, mp_value)
        h = self.model(self._cached_coalesced_graph, h, gamma)
        return h
