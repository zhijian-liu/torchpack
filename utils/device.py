import os

__all__ = ['set_cuda_visible_devices']


def parse_devices(s):
    devs = []
    for dev in s.split(','):
        dev = dev.strip().lower()
        if dev == 'cpu':
            devs.append(-1)
        else:
            if dev.startswith('gpu'):
                dev = dev[3:]
            if '-' in dev:
                l, r = map(int, dev.split('-'))
                devs.extend(range(l, r + 1))
            else:
                devs.append(int(dev))
    return devs


def set_cuda_visible_devices(devs):
    devs = [dev for dev in parse_devices(devs) if dev > -1]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(dev) for dev in devs])
    return devs
