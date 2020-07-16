import os
from typing import List, Union

import torch

__all__ = ['parse_cuda_devices', 'set_cuda_visible_devices']


def parse_cuda_devices(text: str) -> List[int]:
    if text == '*':
        return [device for device in range(torch.cuda.device_count())]

    devices = []
    for device in text.split(','):
        device = device.strip().lower()
        if device == 'cpu':
            continue
        if device.startswith('gpu'):
            device = device[3:]
        if '-' in device:
            l, r = device.split('-')
            devices.extend(range(int(l), int(r) + 1))
        else:
            devices.append(int(device))
    return devices


def set_cuda_visible_devices(devices: Union[str, List[int]],
                             *,
                             environ: os._Environ = os.environ) -> List[int]:
    if isinstance(devices, str):
        devices = parse_cuda_devices(devices)
    environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    return devices
