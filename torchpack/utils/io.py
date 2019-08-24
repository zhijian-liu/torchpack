import json
import torch


__all__ = ['load', 'dump']


def load_pth(f, **kwargs):
    return torch.load(f, **kwargs)


def dump_pth(f, **kwargs):
    return


def dump_json(f, obj):
    with open(f, 'w') as fp:
        json.dump(obj, fp)


def dump(filename, obj):
    with open(filename, 'w') as fp:
        json.dump(obj, fp)


def load(filename):
    with open(filename) as fp:
        return json.load(fp)
