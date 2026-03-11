import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..layers import ConvModule


class FCNHead(nn.Module):
    def __init__(self, c1, c2, num_classes: int = 19):
        super().__init__()
        self.conv = ConvModule(c1, c2, 1)
        self.cls = nn.Conv2d(c2, num_classes, 1)

    def forward(self, features) -> Tensor:
        x = self.conv(features[-1])
        x = self.cls(x)
        return x

