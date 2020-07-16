from typing import List, Tuple, Union

import torch
from torch import nn

from torchpack.models.utils import make_divisible

__all__ = ['MobileNetV2', 'MobileBlockV2']


class MobileBlockV2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 *,
                 stride: int = 1,
                 expansion: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion

        if expansion == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size,
                          stride=stride,
                          padding=kernel_size // 2,
                          groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            mid_channels = round(in_channels * expansion)
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(mid_channels,
                          mid_channels,
                          kernel_size,
                          stride=stride,
                          padding=kernel_size // 2,
                          groups=mid_channels,
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == self.out_channels and self.stride == 1:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    layers: List = [
        32, (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
        (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1), 1280
    ]

    def __init__(self,
                 *,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 width_multiplier: float = 1) -> None:
        super().__init__()

        out_channels = make_divisible(self.layers[0] * width_multiplier, 8)
        layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          3,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )
        ])
        in_channels = out_channels

        for expansion, out_channels, num_blocks, strides in self.layers[1:-1]:
            out_channels = make_divisible(out_channels * width_multiplier, 8)
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(
                    MobileBlockV2(in_channels,
                                  out_channels,
                                  3,
                                  stride=stride,
                                  expansion=expansion))
                in_channels = out_channels

        out_channels = make_divisible(self.layers[-1] * width_multiplier,
                                      8,
                                      min_value=1280)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            ))
        in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
