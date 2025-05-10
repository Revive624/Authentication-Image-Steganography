import torch
import torch.nn as nn
from .rrdb_denselayer import ResidualDenseBlock
from config import cfg


class Inv_hiding_block(nn.Module):
    def __init__(self, secret_num):
        super(Inv_hiding_block, self).__init__()
        self.split_len1 = 3 * 4
        self.split_len2 = 3 * secret_num
        self.clamp = cfg.clamp

        self.r = ResidualDenseBlock(self.split_len1, self.split_len2)
        self.h = ResidualDenseBlock(self.split_len1, self.split_len2)
        self.p = ResidualDenseBlock(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.p(x2)
            y2 = x2 * self.e(self.r(y1)) + self.h(y1)
        else:
            y2 = (x2 - self.h(x1)) / self.e(self.r(x1))
            y1 = x1 - self.p(y2)

        return torch.cat((y1, y2), 1)
   

class Inv_authen_block(nn.Module):
    def __init__(self, channel_num, channel_split_num, cond_num, clamp=1.):
        super(Inv_authen_block, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.cond_num = cond_num

        self.clamp = clamp

        self.F = ResidualDenseBlock(self.split_len2, self.split_len1)
        self.G = ResidualDenseBlock(self.split_len1 + self.cond_num, self.split_len2)
        self.H = ResidualDenseBlock(self.split_len1 + self.cond_num, self.split_len2)

    def forward(self, x, cond, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y1_cond = torch.cat((y1, cond), 1)
            self.s = self.clamp * (torch.sigmoid(self.H(y1_cond)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1_cond)
        else:
            x1_cond = torch.cat((x1, cond), 1)
            self.s = self.clamp * (torch.sigmoid(self.H(x1_cond)) * 2 - 1)
            y2 = (x2 - self.G(x1_cond)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

