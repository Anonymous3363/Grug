
import torch.nn as nn
from . import BaseModel, register_model
from . import HeteroGraphConv
from . import GATConv
@register_model('RGAT')
class RGAT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.hidden_dim,
                   out_dim=args.hidden_dim,
                   h_dim=args.out_dim,
                   etypes=hg.etypes,
                 )

    def __init__(self, in_dim, out_dim, h_dim, etypes):
        super(RGAT, self).__init__()
        self.rel_names = etypes
        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(RGATLayer(
            in_dim, h_dim, self.rel_names))
        self.layers.append(RGATLayer(
            h_dim, out_dim, self.rel_names))
        return

    def forward(self, hg, h_dict, message_):
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict, message_)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict, message_)
        return h_dict


class RGATLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 bias=True,):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv = HeteroGraphConv({
            rel: GATConv(in_feat, out_feat, bias=bias, allow_zero_in_degree=True)
            for rel in rel_names
        })

    def forward(self, g, h_dict, message_):
        h_dict = self.conv(g, h_dict, message_ =message_)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put