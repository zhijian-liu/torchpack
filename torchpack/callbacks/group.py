import time
from collections import defaultdict

import numpy as np
import six
import tensorflow as tf
from tensorpack.tfutils.summary import create_image_summary, create_scalar_summary

from torchpack.callbacks.callback import Callback
from torchpack.callbacks.monitor import Monitor

__all__ = ['CallbackGroup', 'MonitorGroup']


class CallbackGroup(Callback):
    """ A container to hold all callbacks.
    """

    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = callbacks

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def trigger_epoch(self):
        for callback in self.callbacks:
            callback.trigger_epoch()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger(self):
        for callback in self.callbacks:
            callback.trigger()


class MonitorGroup(Monitor):
    """ A container to hold all monitors.
    """

    def __init__(self, monitors):
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
        self.monitors = monitors
        self.scalars = defaultdict(list)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for monitor in self.monitors:
            monitor.set_trainer(trainer)

    def before_train(self):
        for monitor in self.monitors:
            monitor.before_train()

    def after_train(self):
        for monitor in self.monitors:
            monitor.after_train()

    def before_epoch(self):
        for monitor in self.monitors:
            monitor.before_epoch()

    def after_epoch(self):
        for monitor in self.monitors:
            monitor.after_epoch()

    def before_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.after_step(*args, **kwargs)

    def trigger_epoch(self):
        for monitor in self.monitors:
            monitor.trigger_epoch()

    def trigger_step(self):
        for monitor in self.monitors:
            monitor.trigger_step()

    def trigger(self):
        for monitor in self.monitors:
            monitor.trigger()

    def add(self, tag, val):
        for monitor in self.monitors:
            monitor.add(tag, val)

    def add_scalar(self, tag, val):
        if isinstance(val, np.integer):
            val = int(val)
        if isinstance(val, np.floating):
            val = float(val)

        self.scalars[tag].append((self.trainer.global_step, val))

        summary = create_scalar_summary(tag, val)
        for monitor in self.monitors:
            monitor.add_scalar(tag, val)
            monitor.add_summary(summary)

    def add_image(self, tag, val):
        assert isinstance(val, np.ndarray), type(val)

        # todo: double check whether transform is correct
        if val.ndim == 2:
            val = val[np.newaxis, :, :, np.newaxis]
        elif val.ndim == 3:
            if val.shape[-1] in [1, 3, 4]:
                val = val[np.newaxis, ...]
            else:
                val = val[..., np.newaxis]
        assert val.ndim == 4, val.shape

        summary = create_image_summary(tag, val)
        for monitor in self.monitors:
            monitor.add_image(tag, val)
            monitor.add_summary(summary)

    def add_summary(self, summary):
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                for monitor in self.monitors:
                    monitor.add_scalar(val.tag, val.simple_value)

        for monitor in self.monitors:
            monitor.add_summary(summary)

    def add_event(self, event):
        event.step = self.global_step
        event.wall_time = time.time()
        for monitor in self.monitors:
            monitor.add_event(event)

    def get_latest(self, name):
        return self.scalars[name][-1][1]

    def get_history(self, name):
        return self.scalars[name]
