import os

import torch

__all__ = ['parse_devices', 'set_cuda_visible_devices']


def parse_devices(devs):
    gpus = []
    if devs == '*':
        gpus += range(torch.cuda.device_count())
    else:
        for dev in devs.split(','):
            dev = dev.strip().lower()
            if dev != 'cpu':
                if dev.startswith('gpu'):
                    dev = dev[3:]
                if '-' in dev:
                    a, b = map(int, dev.split('-'))
                    gpus += range(a, b + 1)
                else:
                    gpus += [int(dev)]
    return gpus


def set_cuda_visible_devices(devs, env=os.environ):
    devs = parse_devices(devs)
    env['CUDA_VISIBLE_DEVICES'] = ','.join([str(dev) for dev in devs])
    return devs
