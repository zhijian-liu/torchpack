import torch.nn as nn

from torchpack.models.utils import make_divisible

__all__ = ['MobileNetV1', 'MobileBlockV1']


class MobileBlockV1(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride

        super().__init__(
            nn.Conv2d(input_channels, input_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2,
                      groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )


class MobileNetV1(nn.Module):
    blocks = [32, (64, 1, 1), (128, 2, 2), (256, 2, 2), (512, 6, 2), (1024, 2, 2)]

    def __init__(self, num_classes, width_multiplier=1.0):
        super().__init__()
        input_channels = make_divisible(self.blocks[0] * width_multiplier, 8)

        layers = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )]

        for output_channels, num_blocks, strides in self.blocks[1:]:
            output_channels = make_divisible(output_channels * width_multiplier, 8)
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(MobileBlockV1(input_channels, output_channels, 3, stride))
                input_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_channels, num_classes)

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
