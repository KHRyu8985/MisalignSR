import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out += residual
        return out


@ARCH_REGISTRY.register()
class SimpleMWNet(nn.Module):
    def __init__(self):
        super(SimpleMWNet, self).__init__()
        self.downsample1 = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1)
        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.resblock3 = ResidualBlock(128, 256, stride=2)
        self.fc = nn.Linear(256, 1)


    def forward(self, x):
        x = self.downsample1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        x = torch.flatten(x, 1)
        x = self.fc(x)
        weights = F.softmax(x, dim=0)
        weights = weights.squeeze(-1)
        return weights


if __name__ == '__main__':
    num_patches = 16  # Number of patches is the same as the batch size
    model = SimpleMWNet()

    # Create dummy input data with the shape (16, 6, 128, 128)
    dummy_input = torch.randn(16, 6, 128, 128)

    # Forward pass through the network
    output_weights = model(dummy_input)

    # Print the output
    print("Output Weights:", output_weights) # (16)
    print("Output Shape:", output_weights.shape)