from collections import deque

from .container import G

__all__ = ['Config']


class Config(G):
    def __init__(self, func=None, args=None, detach=False, **kwargs):
        super().__init__(**kwargs)

        if func is not None and not callable(func):
            raise TypeError('func "{}" is not a callable function or class'.format(repr(func)))
        if args is not None and not isinstance(args, (list, tuple)):
            raise TypeError('args "{}" is not an iterable tuple or list'.format(repr(args)))

        self._func_ = func
        self._args_ = args
        self._detach_ = detach

    def items(self):
        for key, value in super().items():
            if key not in ['_func_', '_args_', '_detach_']:
                yield key, value

    def keys(self):
        for key, value in self.items():
            yield key

    def values(self):
        for key, value in self.items():
            yield value

    def __call__(self, *args, **kwargs):
        if self._func_ is None:
            return self

        # override args
        if args:
            args = list(args)
        elif self._args_:
            args = list(self._args_)

        # override kwargs
        for key, value in self.items():
            kwargs.setdefault(key, value)

        # recursively call non-detached funcs
        queue = deque([args, kwargs])
        while queue:
            x = queue.popleft()

            if isinstance(x, (list, tuple)):
                items = enumerate(x)
            elif isinstance(x, dict):
                items = x.items()
            else:
                items = []

            for key, value in items:
                if isinstance(value, tuple):
                    value = x[key] = list(value)
                elif isinstance(value, Config):
                    if value._detach_:
                        continue
                    value = x[key] = value()
                queue.append(value)

        return self._func_(*args, **kwargs)

    def __str__(self, indent=0):
        text = ''
        if self._func_ is not None:
            text += ' ' * indent + '[func] = ' + str(self._func_)
            if self._detach_:
                text += '(detach=' + str(self._detach_) + ')'
            text += '\n'
            if self._args_:
                for key, value in enumerate(self._args_):
                    text += ' ' * indent + '[args:' + str(key) + '] = ' + str(value) + '\n'

        for key, value in self.items():
            text += ' ' * indent + '[' + str(key) + ']'
            if isinstance(value, Config):
                text += '\n' + value.__str__(indent + 2)
            else:
                text += ' = ' + str(value)
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
