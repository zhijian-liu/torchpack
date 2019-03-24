import functools
import sys

__all__ = ['get_current_context', 'context_load_inputs', 'context_save_outputs', 'context_save_locals']


class Context(object):
    def __init__(self):
        self.context = dict()

    def update(self, key, val):
        self.context[key] = val

    def get(self, key):
        return self.context[key]


_context = dict()


def get_current_context():
    return _context


def context_load_inputs(*names, context=None):
    if context is None:
        context = get_current_context()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for k in names:
                if k not in kwargs:
                    kwargs[k] = context[k]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def context_save_outputs(*names, context=None):
    if context is None:
        context = get_current_context()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            for k, v in zip(names, res):
                context[k] = v
            return res

        return wrapper

    return decorator


def context_save_locals(*names, context=None):
    if context is None:
        context = get_current_context()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def tracer(frame, event, arg):
                if event == 'return':
                    for k in names:
                        context[k] = frame.f_locals[k]

            sys.setprofile(tracer)
            try:
                res = func(*args, **kwargs)
            finally:
                sys.setprofile(None)
            return res

        return wrapper

    return decorator
