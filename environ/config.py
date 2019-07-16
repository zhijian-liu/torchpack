import importlib.util

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module']


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
        for k, v in self.items():
            if k not in kwargs:
                # instantiate arguments if callable
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


def update_configs_from_module(*modules, recursive=True):
    imported = set()

    def import_module(module):
        if module in imported:
            return
        imported.add(module)

        # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module.split('/')[-1], module)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

    for module in modules:
        if recursive:
            for index in [index for index, char in enumerate(module) if char == '/']:
                import_module(module[:index + 1] + '__init__.py')
        import_module(module)
