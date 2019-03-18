import functools
import sys

__all__ = ['get_current_context', 'torchpack_inputs', 'torchpack_outputs']

_context = dict()


def get_current_context():
    return _context


def torchpack_inputs(keys, prefix='', context=None):
    if context is None:
        context = get_current_context()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for key in keys:
                if key not in kwargs:
                    kwargs[key] = context[prefix + key]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def torchpack_outputs(keys, prefix='', context=None):
    if context is None:
        context = get_current_context()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def tracer(frame, event, arg):
                if event == 'return':
                    for key in keys:
                        context[prefix + key] = frame.f_locals[key]

            sys.setprofile(tracer)
            try:
                res = func(*args, **kwargs)
            finally:
                sys.setprofile(None)
            return res

        return wrapper

    return decorator
