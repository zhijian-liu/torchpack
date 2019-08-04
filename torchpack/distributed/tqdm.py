from tqdm import tqdm as t

from . import get_rank


def tqdm(x, *args, **kwargs):
    if get_rank() == 0:
        return t(x, *args, **kwargs)
    else:
        return x
