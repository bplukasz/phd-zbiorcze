import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def sn(module, enabled=True):
    return spectral_norm(module) if enabled else module

class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else None

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)

        s = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.skip is not None:
            s = self.skip(s)
        return h + s

class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, sn_enabled=True, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = sn(nn.Conv2d(in_ch, out_ch, 3, 1, 1), sn_enabled)
        self.conv2 = sn(nn.Conv2d(out_ch, out_ch, 3, 1, 1), sn_enabled)
        self.skip = sn(nn.Conv2d(in_ch, out_ch, 1, 1, 0), sn_enabled) if in_ch != out_ch else None

    def forward(self, x):
        h = F.relu(x, inplace=True)
        h = self.conv1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        s = x
        if self.downsample:
            s = F.avg_pool2d(s, 2)
        if self.skip is not None:
            s = self.skip(s)
        return h + s

class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=128, resolution=32):
        super().__init__()
        assert resolution in [32, 64]
        self.z_dim = z_dim
        self.resolution = resolution

        self.fc = nn.Linear(z_dim, 4*4*ch*4)
        self.rb1 = ResBlockG(ch*4, ch*2)
        self.rb2 = ResBlockG(ch*2, ch)
        if resolution == 64:
            self.rb3 = ResBlockG(ch, ch)
        self.bn = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.rb1(h)
        h = self.rb2(h)
        if self.resolution == 64:
            h = self.rb3(h)
        h = F.relu(self.bn(h), inplace=True)
        x = torch.tanh(self.conv_out(h))
        return x

class Discriminator(nn.Module):
    def __init__(self, ch=128, resolution=32, sn_enabled=True):
        super().__init__()
        assert resolution in [32, 64]
        self.resolution = resolution

        self.conv_in = sn(nn.Conv2d(3, ch, 3, 1, 1), sn_enabled)
        self.rb1 = ResBlockD(ch, ch, sn_enabled, downsample=True)
        self.rb2 = ResBlockD(ch, ch*2, sn_enabled, downsample=True)
        self.rb3 = ResBlockD(ch*2, ch*4, sn_enabled, downsample=True)
        if resolution == 64:
            self.rb4 = ResBlockD(ch*4, ch*4, sn_enabled, downsample=True)

        self.lin = sn(nn.Linear(ch*4, 1), sn_enabled)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        if self.resolution == 64:
            h = self.rb4(h)
        h = F.relu(h, inplace=True)
        h = h.sum(dim=[2, 3])  # global sum pooling (SNGAN-like)
        out = self.lin(h)
        return out.squeeze(1)

