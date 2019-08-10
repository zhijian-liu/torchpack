import re
import time
from collections import defaultdict

import numpy as np
import six
import tensorflow as tf
from tensorpack.tfutils.summary import create_image_summary, create_scalar_summary
from tensorpack.utils.develop import HIDE_DOC

from torchpack.callbacks.callback import Callback
from torchpack.callbacks.monitor import Monitor

__all__ = ['CallbackGroup', 'MonitorGroup']


class CallbackGroup(Callback):
    """ A container to hold all callbacks, and trigger them iteratively.
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
            callback.trigger_step()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger(self):
        for callback in self.callbacks:
            callback.trigger()


class MonitorGroup(Callback):
    """
    Merge monitors together for trainer to use.
    In training, each trainer will create a :class:`Monitors` instance,
    and you can access it through ``trainer.monitors``.
    You should use ``trainer.monitors`` for logging and it will dispatch your
    logs to each sub-monitor.
    """

    chief_only = False

    def __init__(self, monitors):
        self._scalar_history = ScalarHistory()
        self.monitors = monitors + [self._scalar_history]
        for monitor in self.monitors:
            assert isinstance(monitor, Monitor), type(monitor)

    def set_trainer(self, trainer):
        # scalar_history's other methods were not called.
        # but they are not useful for now
        self._scalar_history.set_trainer(trainer)

    def _dispatch(self, func):
        for m in self.monitors:
            func(m)

    def add_summary(self, summary):
        """
        Put a `tf.Summary`.
        """
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        # TODO other types
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[0-9]+/', '', val.tag)  # TODO move to subclasses

                # TODO This hack is still needed, seem to disappear only when
                # compiled from source.
                suffix = '-summary'  # tensorflow#6150, tensorboard#59
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]

                self._dispatch(lambda m: m.add_scalar(val.tag, val.simple_value))

        self._dispatch(lambda m: m.add_summary(summary))

    def add_scalar(self, name, val):
        """
        Add a scalar.
        """
        if isinstance(val, np.floating):
            val = float(val)
        if isinstance(val, np.integer):
            val = int(val)
        self._dispatch(lambda m: m.add_scalar(name, val))
        s = create_scalar_summary(name, val)
        self._dispatch(lambda m: m.add_summary(s))

    def add_image(self, name, val):
        """
        Add an image.
        Args:
            name (str):
            val (np.ndarray): 2D, 3D (HWC) or 4D (NHWC) numpy array of images
                in range [0,255]. If channel is 3, assumed to be RGB.
        """
        assert isinstance(val, np.ndarray)
        arr = image_to_nhwc(val)
        self._dispatch(lambda m: m.process_image(name, arr))
        s = create_image_summary(name, arr)
        self._dispatch(lambda m: m.add_summary(s))

    def add_event(self, event):
        """
        Add an :class:`tf.Event`.
        `step` and `wall_time` fields of :class:`tf.Event` will be filled automatically.
        Args:
            event (tf.Event):
        """
        event.step = self.global_step
        event.wall_time = time.time()
        self._dispatch(lambda m: m.add_event(event))

    def get_latest(self, name):
        """
        Get latest scalar value of some data.
        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.
        Returns:
            scalar
        """
        return self._scalar_history.get_latest(name)[1]

    def get_history(self, name):
        """
        Get a history of the scalar value of some data.
        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.
        Returns:
            a list of (global_step, value) pairs: history data for this scalar
        """
        return self._scalar_history.get_history(name)


def image_to_nhwc(arr):
    if arr.ndim == 4:
        pass
    elif arr.ndim == 3:
        if arr.shape[-1] in [1, 3, 4]:
            arr = arr[np.newaxis, :]
        else:
            arr = arr[:, :, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError("Array of shape {} is not an image!".format(arr.shape))
    return arr


class ScalarHistory(Monitor):
    """
    Only internally used by monitors.
    """

    def __init__(self):
        self._hist = defaultdict(list)

    @HIDE_DOC
    def add_scalar(self, name, val):
        self._hist[name].append((self.trainer.global_step, float(val)))

    def get_latest(self, name):
        hist = self._hist[name]
        if len(hist) == 0:
            raise KeyError("No available data for the key: {}".format(name))
        else:
            return hist[-1]

    def get_history(self, name):
        return self._hist[name]
