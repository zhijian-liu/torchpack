import torch.nn as nn

from ..utils import make_divisible

__all__ = ['MobileNetV2', 'MobileBlockV2']


class MobileBlockV2(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 expansion=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion

        if expansion == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels,
                          input_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=kernel_size // 2,
                          groups=input_channels,
                          bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            expansion_channels = round(input_channels * expansion)
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels,
                          expansion_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(expansion_channels,
                          expansion_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=kernel_size // 2,
                          groups=expansion_channels,
                          bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(expansion_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, x):
        if self.input_channels == self.output_channels and self.stride == 1:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    blocks = [
        32, (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
        (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1), 1280
    ]

    def __init__(self, input_channels=3, num_classes=1000, width_multiplier=1):
        super().__init__()

        output_channels = make_divisible(self.blocks[0] * width_multiplier, 8)
        layers = [
            nn.Sequential(
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU6(inplace=True),
            )
        ]
        input_channels = output_channels

        for expansion, output_channels, num_blocks, strides in \
                self.blocks[1:-1]:
            output_channels = make_divisible(
                output_channels * width_multiplier, 8)
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(
                    MobileBlockV2(input_channels,
                                  output_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  expansion=expansion))
                input_channels = output_channels

        output_channels = make_divisible(self.blocks[-1] * width_multiplier, 8, \
                                         min_value=1280)
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU6(inplace=True),
            ))
        input_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
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

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
