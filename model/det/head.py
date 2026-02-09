import torch
import torch.nn as nn
from model.det.layers import ConvBnRelu

class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super(DBHead, self).__init__()
        self.k = k

        # Binary Map Branch
        self.bin_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, 3, 1, 1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )

        # Threshold Map Branch
        self.thresh_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, 3, 1, 1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )

    def step_function(self, x, y):
        # Differentiable Binarization
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        # x: (N, inner_channels, H/4, W/4)
        binary = self.bin_conv(x)
        thresh = self.thresh_conv(x)
        thresh_binary = self.step_function(binary, thresh)

        return {'binary': binary, 'thresh': thresh, 'thresh_binary': thresh_binary}
