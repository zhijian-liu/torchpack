import copy

__all__ = ['split_trainval']


def split_trainval(dataset, ratio=0.5):
    size = len(dataset['train'])

    dataset['valid'] = copy.deepcopy(dataset['train'])
    return dataset
