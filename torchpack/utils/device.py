import os

import torch

__all__ = ['parse_cuda_devices', 'set_cuda_visible_devices']


def parse_cuda_devices(text):
    if text == '*':
        return range(torch.cuda.device_count())

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


def set_cuda_visible_devices(devices, *, environ=os.environ):
    if isinstance(devices, str):
        devices = parse_cuda_devices(devices)
    environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    return devices
