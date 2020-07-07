import fnmatch

__all__ = ['NameMatcher']


class NameMatcher:
    def __init__(self, patterns):
        if patterns is None:
            patterns = []
        elif isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = patterns

    def match(self, name):
        for pattern in self.patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
