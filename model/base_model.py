from abc import ABCMeta
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    def extra_loss(self):
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        raise NotImplementedError
