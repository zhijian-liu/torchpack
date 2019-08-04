import os

import torch

__all__ = ['set_cuda_visible_devices']


def set_cuda_visible_devices(devs, env=os.environ):
    gpus = []
    if devs == '*':
        gpus += range(torch.cuda.device_count())
    else:
        gpus = []
        for dev in devs.split(','):
            dev = dev.strip().lower()
            if dev == 'cpu':
                continue
            if dev.startswith('gpu'):
                dev = dev[3:]
            if '-' in dev:
                l, r = map(int, dev.split('-'))
                gpus += range(l, r + 1)
            else:
                gpus += [int(dev)]
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
    return gpus
