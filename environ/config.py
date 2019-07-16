import importlib.util
import os

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module', 'update_configs_from_options']


class Config(G):
    def __init__(self, callable=None):
        super().__init__()
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

        # instantiate if callable
        for k, v in self.items():
            if k not in kwargs:
                kwargs[k] = v() if isinstance(v, Config) else v
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


def update_configs_from_options(opts):
    index = 1 if opts[0] == '--' else 0
    while index < len(opts):
        opt = opts[index]
        if opts[0] != '--' and opt.startswith('--configs.'):
            opt = opt.replace('--configs.', '')
        elif opts[0] == '--' and opt.startswith('configs.'):
            opt = opt.replace('configs.', '')
        else:
            raise Exception('unrecognized options "{}"'.format(opt))

        if '=' in opt:
            keys, val, index = opt[:opt.index('=')], opt[opt.index('=') + 1:], index + 1
        else:
            keys, val, index = opts[index], opts[index + 1], index + 2

        keys = keys.split('.')
        if val.startswith('int{') and val.endswith('}'):
            val = int(val[4:-1])
        elif val.startswith('float{') and val.endswith('}'):
            val = float(val[6:-1])

        o = configs
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                o[k] = val
            elif k not in o:
                o[k] = Config()
            o = o[k]
