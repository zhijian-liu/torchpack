import fnmatch
from typing import List, Optional, Union

__all__ = ['NameMatcher']


class NameMatcher:

    def __init__(self, patterns: Optional[Union[str, List[str]]]) -> None:
        if patterns is None:
            patterns = []
        elif isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = patterns

    def match(self, name: str) -> bool:
        for pattern in self.patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
