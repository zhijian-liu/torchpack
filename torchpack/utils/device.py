import os

import torch

__all__ = ['parse_devices', 'set_cuda_visible_devices']


def parse_devices(devs):
    if devs == '*':
        return range(torch.cuda.device_count())

    gpus = []
    for dev in devs.split(','):
        dev = dev.strip().lower()
        if dev == 'cpu':
            continue
        if dev.startswith('gpu'):
            dev = dev[3:]
        if '-' in dev:
            l, r = dev.split('-')
            gpus += range(int(l), int(r) + 1)
        else:
            gpus += [int(dev)]
    return gpus


def set_cuda_visible_devices(devs, env=os.environ):
    devs = parse_devices(devs)
    env['CUDA_VISIBLE_DEVICES'] = ','.join([str(dev) for dev in devs])
    return devs
