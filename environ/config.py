import importlib.util
import os

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module', 'update_configs_from_arguments']


class Config(G):
    def __init__(self, callable=None, **kwargs):
        super().__init__(**kwargs)
        self.__callable__ = callable

    def keys(self):
        for k in super().keys():
            if k != '__callable__':
                yield k

    def items(self):
        for k, v in super().items():
            if k != '__callable__':
                yield k, v

    def __call__(self, *args, **kwargs):
        if self.__callable__ is None:
            return self

        # override kwargs: call v() if callable
        for k, v in self.items():
            if k not in kwargs:
                if isinstance(v, Config):
                    kwargs[k] = v()
                else:
                    kwargs[k] = v

        # call/instantiate with args and kwargs
        return self.__callable__(*args, **kwargs)

    def __str__(self, indent=0, verbose=None):
        # default value: True for non-callable; False for callable
        verbose = (self.__callable__ is None) if verbose is None else verbose

        assert self.__callable__ is not None or verbose
        if self.__callable__ is not None and not verbose:
            return str(self.__callable__)

        text = ''
        if self.__callable__ is not None and indent == 0:
            text += str(self.__callable__) + '\n'
            indent += 2

        for k, v in self.items():
            text += ' ' * indent + '[{}]'.format(k)
            if not isinstance(v, Config):
                text += ' = {}'.format(v)
            else:
                if v.__callable__ is not None:
                    text += ' = ' + str(v.__callable__)
                text += '\n' + v.__str__(indent + 2, verbose=verbose)
            text += '\n'

        # remove the last newline
        return text[:-1]


configs = Config()


def update_configs_from_module(*modules):
    imported_modules = set()

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    def import_module(module):
        if module not in imported_modules:
            spec = importlib.util.spec_from_file_location(module.split('/')[-1], module)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            imported_modules.add(module)

    for m in modules:
        for k, c in enumerate(m):
            if c == '/' and os.path.exists(m[:k + 1] + '__init__.py'):
                import_module(m[:k + 1] + '__init__.py')
        import_module(m)


def parse(x):
    # str
    if (x[0] == '\'' and x[-1] == '\'') or (x[0] == '\"' and x[-1] == '\"'):
        return x[1:-1]

    # int / float
    try:
        x = eval(x)
    except:
        pass
    else:
        return x

    # str
    return x


def update_configs_from_arguments(args):
    index = 0
    while index < len(args):
        arg = args[index]

        if arg.startswith('--configs.'):
            arg = arg.replace('--configs.', '')
        else:
            raise Exception('unrecognized argument "{}"'.format(arg))

        if '=' not in arg:
            index, ks, v = index + 2, arg.split('.'), parse(args[index + 1])
        else:
            index, ks, v = index + 1, arg[:arg.index('=')].split('.'), parse(arg[arg.index('=') + 1:])

        o = configs
        for k in ks[:-1]:
            if k not in o:
                o[k] = Config()
            o = o[k]
        o[ks[-1]] = v
