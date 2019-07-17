import numpy as np
import torch.nn as nn

__all__ = ['flops_handlers']


def conv(module, inputs, outputs):
    kernel_size = module.weight.size()
    output_size = outputs.size()
    return np.prod([output_size[0]] + list(output_size[2:]) + list(kernel_size))


def gemm(module, inputs, outputs):
    kernel_size = module.weight.size()
    output_size = outputs.size()
    return np.prod([output_size[0]] + list(kernel_size))


flops_handlers = [
    (nn.Linear, gemm),
    ((nn.Conv1d, nn.Conv2d, nn.Conv3d), conv),
    ((nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d), conv),

    ((nn.ReLU, nn.ReLU6), None),
    ((nn.Dropout, nn.Dropout2d, nn.Dropout3d), None),
    ((nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d), None),
    ((nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d), None),
    ((nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), None),
    ((nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d), None),
    ((nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d), None)
]
