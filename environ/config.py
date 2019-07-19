import collections
import importlib.util
import os
from collections import deque

import six

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module', 'update_configs_from_arguments']


class Config(G):
    def __init__(self, func=None, args=None, **kwargs):
        super().__init__(**kwargs)
        self._func_ = func
        self._args_ = args

    def items(self):
        for k, v in super().items():
            if k not in ['_func_', '_args_']:
                yield k, v

    def keys(self):
        for k, v in self.items():
            yield k

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
            if k not in kwargs:
                kwargs[k] = v

        # call all funcs in a recursive manner
        queue = deque([args, kwargs])
        while queue:
            x = queue.popleft()

            if not isinstance(x, six.string_types) and isinstance(x, (collections.Sequence, collections.UserList)):
                children = enumerate(x)
            elif isinstance(x, Config):
                children = x.__dict__.items()
            elif isinstance(x, (collections.Mapping, collections.UserDict)):
                children = x.items()
            else:
                children = []

            for k, v in children:
                if isinstance(v, Config):
                    v = x[k] = v()
                if isinstance(v, tuple):
                    v = x[k] = list(v)
                queue.append(v)

        return self._func_(*args, **kwargs)

    def __str__(self, indent=0, verbose=None):
        # default value: True for non-callable; False for callable
        verbose = (self._func_ is None) if verbose is None else verbose

        assert self._func_ is not None or verbose
        if self._func_ is not None and not verbose:
            return str(self._func_)

        text = ''
        if self._func_ is not None and indent == 0:
            text += str(self._func_) + '\n'
            indent += 2

        for k, v in self.items():
            text += ' ' * indent + '[{}]'.format(k)
            if not isinstance(v, Config):
                text += ' = {}'.format(v)
            else:
                if v._func_ is not None:
                    text += ' = ' + str(v._func_)
                text += '\n' + v.__str__(indent + 2, verbose=verbose)
            text += '\n'

        # remove the last newline
        return text[:-1]


configs = Config()


def update_configs_from_module(*modules):
    imported_modules = set()

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    def exec_module_once(mod):
        if mod in imported_modules:
            return
        imported_modules.add(mod)
        spec = importlib.util.spec_from_file_location(os.path.basename(mod), mod)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

    for module in modules:
        module = os.path.normpath(module)
        for index, char in enumerate(module):
            if index == 0 or char == os.sep:
                submod = os.path.join(module[:index], '__init__.py')
                if os.path.exists(submod):
                    exec_module_once(submod)
        exec_module_once(module)


def update_configs_from_arguments(args):
    index = 0

    while index < len(args):
        arg = args[index]

        if arg.startswith('--configs.'):
            arg = arg.replace('--configs.', '')
        else:
            raise Exception('unrecognized argument "{}"'.format(arg))

        if '=' not in arg:
            index, keys, val = index + 2, arg.split('.'), args[index + 1]
        else:
            index, keys, val = index + 1, arg[:arg.index('=')].split('.'), arg[arg.index('=') + 1:]

        config = configs
        for k in keys[:-1]:
            if k not in config:
                config[k] = Config()
            config = config[k]

        def parse(x):
            if (x[0] == '\'' and x[-1] == '\'') or (x[0] == '\"' and x[-1] == '\"'):
                return x[1:-1]
            try:
                x = eval(x)
            except:
                pass
            return x

        config[keys[-1]] = parse(val)
