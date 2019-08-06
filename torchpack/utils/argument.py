import argparse

from torchpack.utils.config import Config
from torchpack.utils.device import set_cuda_visible_devices
from torchpack.utils.imp import load_source

__all__ = ['ArgumentParser']


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('type', Config, _config)
        self.register('action', 'set_devices', SetDeviceAction)
        self.register('action', 'parse_configs', ParseConfigFile)


def _config(string):
    configs = load_source(string).configs
    configs.cfg_file = string
    return configs


# from https://github.com/vacancy/Jacinle/blob/master/jacinle/cli/argument.py
class SetDeviceAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, set_cuda_visible_devices(values))


class ParseConfigFile(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'cfg_file', values)
        setattr(namespace, self.dest, load_source(values).configs)

# def parse(x):
#     if (x[0] == '\'' and x[-1] == '\'') or (x[0] == '\"' and x[-1] == '\"'):
#         return x[1:-1]
#     try:
#         x = eval(x)
#     except:
#         pass
#     return x
#
# def parse_args(self, args=None, namespace=None):
#     index = 0
#
#     while index < len(opts):
#         arg = opts[index]
#
#         if arg.startswith('--configs.'):
#             arg = arg.replace('--configs.', '')
#         else:
#             raise Exception('unrecognized argument "{}"'.format(arg))
#
#         if '=' in arg:
#             index, keys, val = index + 1, arg[:arg.index('=')].split('.'), arg[arg.index('=') + 1:]
#         else:
#             index, keys, val = index + 2, arg.split('.'), opts[index + 1]
#
#         config = configs
#         for k in keys[:-1]:
#             if k not in config:
#                 config[k] = Config()
#             config = config[k]
#         config[keys[-1]] = parse(val)
#     return args, configs
