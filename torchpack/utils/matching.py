import fnmatch

import six

__all__ = ['NameMatcher', 'IENameMatcher']


class NameMatcher:
    def __init__(self, patterns):
        if patterns is None:
            patterns = []
        elif isinstance(patterns, six.string_types):
            patterns = [patterns]
        self.patterns = patterns

    def match(self, name):
        for pattern in self.patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False


class IENameMatcher:
    def __init__(self, includes, excludes):
        self.includes = NameMatcher(includes)
        self.excludes = NameMatcher(excludes)

    def match(self, name):
        return self.includes.match(name) and not self.excludes.match(name)
