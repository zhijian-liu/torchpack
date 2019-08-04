from abc import abstractmethod

__all__ = ['Meter']


class Meter(object):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *inputs):
        pass

    @abstractmethod
    def compute(self):
        pass
