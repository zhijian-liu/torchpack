import torch

import torch.nn as nn

__all__ = ['ShuffleNetV2', 'ShuffleBlockV2']


def shuffle_channel(inputs, groups):
    batch_size, num_channels, height, width = inputs.size()
    return inputs.view(batch_size, groups, num_channels // groups, height, width) \
        .transpose(1, 2).contiguous().view(batch_size, num_channels, height, width)


class ShuffleBlockV2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if stride == 1:
            input_channels = input_channels // 2
        output_channels = output_channels // 2

        if stride != 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2,
                          groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),

                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2,
                      groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),

            nn.Conv2d(output_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x[:, :x.size(1) // 2], self.branch2(x[:, x.size(1) // 2:])
        else:
            x1, x2 = self.branch1(x), self.branch2(x)
        return shuffle_channel(torch.cat([x1, x2], dim=1), groups=2)


class ShuffleNetV2(nn.Module):
    blocks = {0.5: [24, (48, 4, 2), (96, 8, 2), (192, 4, 2), 1024],
              1.0: [24, (116, 4, 2), (232, 8, 2), (464, 4, 2), 1024],
              1.5: [24, (116, 4, 2), (232, 8, 2), (464, 4, 2), 1024],
              2.0: [24, (244, 4, 2), (488, 8, 2), (976, 4, 2), 2048]}

    def __init__(self, num_classes, width_multiplier=1.0):
        super().__init__()
        input_channels = self.blocks[width_multiplier][0]
        last_channels = self.blocks[width_multiplier][-1]

        layers = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )]

        for output_channels, num_blocks, strides in self.blocks[width_multiplier][1:-1]:
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(ShuffleBlockV2(input_channels, output_channels, 3, stride))
                input_channels = output_channels

        layers.append(nn.Sequential(
            nn.Conv2d(input_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU(inplace=True)
        ))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(last_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
