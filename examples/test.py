from torchpack.utils.context import *


@torchpack_outputs(['inputs', 'targets'])
def load_data():
    inputs = 1
    targets = 2


@torchpack_inputs(['inputs', 'targets'])
def print_data(inputs, targets):
    print(inputs, targets)


load_data()
print_data()
