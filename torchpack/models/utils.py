__all__ = ['make_divisible']


# from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def make_divisible(v, divisor, *, min_value=None):
    if min_value is None:
        min_value = divisor
    x = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if x < 0.9 * v:
        x += divisor
    return x
