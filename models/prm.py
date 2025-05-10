import torch.nn as nn
from modules.module_util import initialize_weights


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class PredictiveModule(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=8):
        super(PredictiveModule, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        residual_block = []
        for i in range(block_num_rbm):
            residual_block.append(ResidualBlockNoBN(nf))
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        x = self.conv_in(x)
        res = self.residual_block(x)
        return res
