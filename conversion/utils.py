import torch
import torch.nn as nn
import torch.nn.functional as F

class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x+3, 0., 6.) / 6.

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x//2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self, x):
        return self.act(self.conv(x))