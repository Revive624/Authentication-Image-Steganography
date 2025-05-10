import torch
import torch.nn as nn
import config as c
from .hinet import Invertible_hiding, Invertible_authen
from .prm import PredictiveModule
import modules.Unet_common as common
from .unet import Unet_down


class Model_hiding(nn.Module):
    def __init__(self, device, channel=3, secret_num=1):
        super(Model_hiding, self).__init__()
        self.channel = channel
        self.inn_hiding = Invertible_hiding(secret_num)
        self.dwt = common.DWT()
        self.iwt = common.IWT()
        self.prm = PredictiveModule(4 * channel, secret_num * channel)
        self.device = device

    def forward(self, cover, secrets_or_num, rev=False):
        if not rev:
            cover_dwt = self.dwt(cover)
            cover_low = cover_dwt[:, 0: self.channel]
            secrets = torch.cat(secrets_or_num, 1)
            res = self.inn_hiding(torch.cat((cover_dwt, secrets), 1))
            stego = self.iwt(res[:, 0: 4 * self.channel])
            stego_low = res[:, 0: 1 * self.channel]
            r_o = res[:, 4 * self.channel:]
            return stego, cover_low,stego_low,r_o
        else:
            reveals_low = []
            stego_dwt = self.dwt(cover)
            r_p = self.prm(stego_dwt)
            res = self.inn_hiding(torch.cat((stego_dwt, r_p), 1), rev=True)[:, 4 * self.channel:]
            for i in range(secrets_or_num):
                reveal_low = res[:, i * 3: (i + 1) * 3]
                reveals_low.append(reveal_low)
            return reveals_low, r_p


class Model_authen(nn.Module):
    def __init__(self, device, channel=3, cond_num=1):
        super(Model_authen, self).__init__()
        self.channel = channel
        self.model = Invertible_authen(cond_num=cond_num)
        self.device = device

    def forward(self, gt, cond, rev=False):
        if not rev:
            res = self.model(gt, cond)
            low = res[:, 0: self.channel]
            high = res[:, self.channel:]
            return low, high
        else:
            B, _, H, W = gt.shape
            z = torch.randn((B, 9, H, W)).to(self.device)
            res = self.model(torch.cat((gt, z), 1), cond, rev=True)
            return res


class Model_lock(nn.Module):
    def __init__(self, device, channel=3, cond_num=1):
        super(Model_lock, self).__init__()
        self.channel = channel
        self.device = device
        self.cond_num = cond_num
        self.cover_lock_gen = Unet_down()
        self.cover_key_gen = Unet_down()

    def forward(self, x, mode):
        if mode == 'cover_lock':
            return self.cover_lock_gen(x)
        elif mode == 'secret_lock':
            return self.cover_lock_gen(x, no_down=True)
        elif mode == 'cover_key':
            return self.cover_key_gen(x)
        elif mode == 'secret_key':
            return self.cover_key_gen(x, no_down=True)


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
