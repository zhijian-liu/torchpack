import collections
from collections import deque

import six

from .container import G

__all__ = ['Config']


class Config(G):
    def __init__(self, func=None, args=None, detach=False, **kwargs):
        if func is not None and not callable(func):
            raise TypeError('func "{}" is not a callable function or class'.format(repr(func)))
        if args is not None and not isinstance(args, (collections.Sequence, collections.UserList)):
            raise TypeError('args "{}" is not an iterable tuple or list'.format(repr(args)))

        super().__init__(**kwargs)
        self._func_ = func
        self._args_ = args
        self._detach_ = detach

    def items(self):
        for k, v in super().items():
            if k not in ['_func_', '_args_', '_detach_']:
                yield k, v

    def keys(self):
        for k, v in self.items():
            yield k

    def values(self):
        for k, v in self.items():
            yield v

    def __call__(self, *args, **kwargs):
        if self._func_ is None:
            return self

        # override args
        if args:
            args = list(args)
        elif self._args_:
            args = list(self._args_)

        # override kwargs
        for k, v in self.items():
            kwargs.setdefault(k, v)

        # recursively call non-detached funcs
        queue = deque([args, kwargs])
        while queue:
            x = queue.popleft()

            if isinstance(x, (collections.Sequence, collections.UserList)) and not isinstance(x, six.string_types):
                items = enumerate(x)
            elif isinstance(x, (collections.Mapping, collections.UserDict)):
                items = x.items()
            else:
                items = []

            for k, v in items:
                if isinstance(v, tuple):
                    v = x[k] = list(v)
                elif isinstance(v, Config):
                    if v._detach_:
                        continue
                    v = x[k] = v()
                queue.append(v)

        return self._func_(*args, **kwargs)

    def __str__(self, indent=0):
        text = ''
        if self._func_ is not None:
            text += ' ' * indent + '[func] = ' + str(self._func_)
            if self._detach_:
                text += '(detach=' + str(self._detach_) + ')'
            text += '\n'
            if self._args_:
                for k, v in enumerate(self._args_):
                    text += ' ' * indent + '[args:' + str(k) + '] = ' + str(v) + '\n'

        for k, v in self.items():
            text += ' ' * indent + '[' + str(k) + ']'
            if isinstance(v, Config):
                text += '\n' + v.__str__(indent + 2)
            else:
                text += ' = ' + str(v)
            text += '\n'

        while text and text[-1] == '\n':
            text = text[:-1]
        return text

    def __repr__(self):
        text = ''
        if self._func_ is not None:
            text += repr(self._func_)

        items = []
        if self._func_ is not None and self._args_:
            items += [repr(v) for v in self._args_]
        items += [str(k) + '=' + repr(v) for k, v in self.items()]
        if self._func_ is not None and self._detach_:
            items += ['detach=' + repr(self._detach_)]

        text += '(' + ', '.join(items) + ')'
        return text
