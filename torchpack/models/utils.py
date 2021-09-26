from typing import Optional

__all__ = ['make_divisible']


# from https://tinyurl.com/vke23tt5
def make_divisible(
    v: int,
    divisor: int,
    *,
    min_value: Optional[int] = None,
) -> int:
    if min_value is None:
        min_value = divisor
    x = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if x < 0.9 * v:
        x += divisor
    return x
