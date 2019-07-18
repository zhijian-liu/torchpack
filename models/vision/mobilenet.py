import torch
import torch.nn as nn

__all__ = ['MobileNet']


# conv with batch normalization
def conv_bn(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


# depth-wise conv
def conv_dw(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel, bias=False),
        nn.BatchNorm2d(input_channel),
        nn.ReLU(inplace=True),

        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


class MobileNetref(nn.Module):
    def __init__(self):
        super(MobileNetref, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.feature = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

class MobileNet(nn.Module):
    """
    MobileNet class
    """
    blocks_dw = [
        (32, 64, 1),
        (64, 128, 2),
        (128, 128, 1),
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 1024, 2),
        (1024, 1024, 1)
    ]

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        layers = [conv_bn(3, 32, 2)]

        for layer_param in self.blocks_dw:
            layers.append(conv_dw(*layer_param))

        layers.append(nn.AvgPool2d(7))

        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNet(num_classes=1000)
    model2 = MobileNetref()
    print(model)

    x = torch.randn(10, 3, 224, 224, device=torch.device('cpu'), dtype = torch.float)
    torch.save(model.state_dict(), './model.pth')

    model2.load_state_dict(torch.load('./model.pth'))
    model2.eval()

    model.eval()
    y = model(x)

    y2 = model2(x)

    # y2.load(y.di)

    # print(model.state_dict())
    # print(model2.state_dict())

    # print(min((y==y2).flatten()))

    print(y)
    print(y2)