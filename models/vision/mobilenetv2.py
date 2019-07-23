import torch.nn as nn

__all__ = ['MobileNetV2', 'MobileBlockV2']


class MobileBlockV2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio

        if expand_ratio == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2,
                          groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU6(inplace=True),

                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        else:
            expand_channels = round(input_channels * expand_ratio)

            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, expand_channels, 1, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),

                nn.Conv2d(expand_channels, expand_channels, kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2,
                          groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),

                nn.Conv2d(expand_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        if self.input_channels == self.output_channels and self.stride == 1:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    blocks = [32, (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
              (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1), 1280]

    def __init__(self, num_classes, width_multiplier=1.0):
        super().__init__()
        input_channels = round(self.blocks[0] * width_multiplier)
        last_channels = round(self.blocks[-1] * max(width_multiplier, 1.0))

        layers = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )]

        for expand_ratio, output_channels, num_blocks, strides in self.blocks[1:-1]:
            output_channels = round(output_channels * width_multiplier)
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(MobileBlockV2(input_channels, output_channels, 3, stride, expand_ratio))
                input_channels = output_channels

        layers.append(nn.Sequential(
            nn.Conv2d(input_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace=True)
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
