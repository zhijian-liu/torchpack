__all__ = ['naturaldelta']


def naturaldelta(seconds: float) -> str:
    if seconds < 0:
        raise ValueError('`seconds` needs to be >= 0.')
    elif seconds == 0:
        return '0 second'

    if seconds < 60:
        values = [0, 0, 0, seconds]
    else:
        values = []
        for divisor in [86400, 3600, 60, 1]:
            value = int(seconds // divisor)
            seconds -= value * divisor
            values.append(value)

    texts = []
    for k, unit in enumerate(['day', 'hour', 'minute', 'second']):
        if values[k] == 0:
            continue
        if values[k] > 1:
            unit += 's'
        texts.append(f'{values[k]:.3g} {unit}')
    return ' '.join(texts)
