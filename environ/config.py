import importlib.util

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module', 'update_configs_from_arguments']


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

        # instantiate arguments if callable
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


def update_configs_from_module(*paths):
    imported_modules = set()

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    def import_module(module):
        if module not in imported_modules:
            spec = importlib.util.spec_from_file_location(module.split('/')[-1], module)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            imported_modules.add(module)

    for path in paths:
        for index in [index for index, char in enumerate(path) if char == '/']:
            import_module(path[:index + 1] + '__init__.py')
        import_module(path)


def update_configs_from_arguments(opts):
    for opt in opts:
        if not opt.startswith('--configs.'):
            continue

        opt = opt.replace('--configs.', '')

        index = opt.index('=')
        a = opt[:index]
        b = opt[index + 1:]

        if b.startswith('float'):
            b = float(b[6:-1])

        current = configs
        for k in a.split('.')[:-1]:
            current = current[k]
        current[a.split('.')[-1]] = b
