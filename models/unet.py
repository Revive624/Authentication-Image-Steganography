import torch
import torch.nn as nn


class Unet_down(nn.Module):
    def __init__(self):
        super(Unet_down, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

    def forward(self, x, no_down=False):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)
        if no_down:
            down1 = x2
        else:
            down1 = self.maxpool_1(x2)
        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        out = self.relu2_2(x4)
        if no_down:
            return out + x
        else:
            return out + self.maxpool_1(x)


if __name__ == '__main__':
    input_data = torch.randn([1, 3, 256, 256])
    unet = Unet_down()
    trainable_num = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(trainable_num)
