import sys

from .. import distributed as dist

__all__ = ['tqdm', 'trange']


def tqdm(iterable=None,
         desc=None,
         total=None,
         leave=True,
         file=sys.stdout,
         ncols=None,
         mininterval=0.1,
         maxinterval=10.0,
         miniters=None,
         ascii=None,
         disable=False,
         unit='it',
         unit_scale=False,
         dynamic_ncols=False,
         smoothing=0.3,
         bar_format=None,
         initial=0,
         position=None,
         postfix=None,
         unit_divisor=1000,
         write_bytes=None,
         gui=False,
         master_only=True,
         **kwargs):
    from tqdm import tqdm

    if master_only:
        disable = disable or not dist.is_master()

    return tqdm(iterable=iterable,
                desc=desc,
                total=total,
                leave=leave,
                file=file,
                ncols=ncols,
                mininterval=mininterval,
                maxinterval=maxinterval,
                miniters=miniters,
                ascii=ascii,
                disable=disable,
                unit=unit,
                unit_scale=unit_scale,
                dynamics_ncols=dynamic_ncols,
                smoothing=smoothing,
                bar_format=bar_format,
                initial=initial,
                position=position,
                postfix=postfix=,
                unit_divisor=unit_divisor,
                write_bytes=write_bytes,
                gui=gui,
                **kwargs)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
