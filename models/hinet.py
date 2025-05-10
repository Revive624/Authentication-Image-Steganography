from .invblock import Inv_hiding_block, Inv_authen_block
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Invertible_hiding(nn.Module):

    def __init__(self, secret_num=1):
        super(Invertible_hiding, self).__init__()

        self.inv1 = Inv_hiding_block(secret_num)
        self.inv2 = Inv_hiding_block(secret_num)
        self.inv3 = Inv_hiding_block(secret_num)
        self.inv4 = Inv_hiding_block(secret_num)
        self.inv5 = Inv_hiding_block(secret_num)
        self.inv6 = Inv_hiding_block(secret_num)
        self.inv7 = Inv_hiding_block(secret_num)
        self.inv8 = Inv_hiding_block(secret_num)

        self.inv9 = Inv_hiding_block(secret_num)
        self.inv10 = Inv_hiding_block(secret_num)
        self.inv11 = Inv_hiding_block(secret_num)
        self.inv12 = Inv_hiding_block(secret_num)
        self.inv13 = Inv_hiding_block(secret_num)
        self.inv14 = Inv_hiding_block(secret_num)
        self.inv15 = Inv_hiding_block(secret_num)
        self.inv16 = Inv_hiding_block(secret_num)

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, cond=None, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac
    

class Invertible_authen(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, block_num=[8], down_num=1, cond_num=1):
        super(Invertible_authen, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = Inv_authen_block(current_channel, channel_out, cond_num)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, cond, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.forward(out, cond, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, cond, rev)

        return out
    