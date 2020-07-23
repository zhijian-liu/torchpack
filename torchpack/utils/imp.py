import importlib.util
import os
from types import ModuleType
from typing import Optional

__all__ = ['load_source']


def load_source(fpath: str, *, name: Optional[str] = None) -> ModuleType:
    if name is None:
        name = os.path.basename(fpath)
        if name.endswith('.py'):
            name = name[:-3]
        name = name.replace('.', '_')

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    spec = importlib.util.spec_from_file_location(name, fpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
