import sys

__all__ = ['tqdm', 'trange']


def tqdm(iterable, **kwargs):
    from tqdm.auto import tqdm
    return tqdm(iterable, **kwargs, file=sys.stdout)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
