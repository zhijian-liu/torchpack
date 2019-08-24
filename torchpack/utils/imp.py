import importlib
import os

__all__ = ['load_source']


def load_source(filename, name=None):
    if name is None:
        name = os.path.basename(filename)
        if name.endswith('.py'):
            name = name[:-3]
        name = name.replace('.', '_')

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    spec = importlib.util.spec_from_file_location(name, filename)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo
