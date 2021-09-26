import importlib.util
import os
from types import ModuleType
from typing import Optional

__all__ = ['load_source']


def load_source(
    fpath: str,
    *,
    name: Optional[str] = None,
) -> Optional[ModuleType]:
    if name is None:
        name = os.path.basename(fpath)
        if name.endswith('.py'):
            name = name[:-3]
        name = name.replace('.', '_')

    # from https://tinyurl.com/4d23vmm9
    spec = importlib.util.spec_from_file_location(name, fpath)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(module)  # type: ignore [attr-defined]
    return module
