import argparse

from torchpack.utils.device import set_cuda_visible_devices

__all__ = ['ArgumentParser']


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('action', 'set_devices', SetDeviceAction)


# from https://github.com/vacancy/Jacinle/blob/master/jacinle/cli/argument.py
class SetDeviceAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, set_cuda_visible_devices(values))
