__all__ = ['G']


# from https://github.com/vacancy/Jacinle/blob/master/jacinle/utils/container.py
class G(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
