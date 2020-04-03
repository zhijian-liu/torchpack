import os.path as osp

import torchpack.utils.fs as fs

__all__ = ['Callback', 'LambdaCallback', 'ProxyCallback', 'Callbacks']


class Callback:
    """
    Base class for all callbacks.
    """
    master_only = False

    def set_trainer(self, trainer):
        self.trainer = trainer
        if self.trainer.is_master or not self.master_only:
            self._set_trainer(trainer)

    def _set_trainer(self, trainer):
        pass

    def before_train(self):
        if self.trainer.is_master or not self.master_only:
            self._before_train()

    def _before_train(self):
        """
        Called before training.
        """
        pass

    def before_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._before_epoch()

    def _before_epoch(self):
        """
        Called before every epoch.
        """
        pass

    def before_step(self, feed_dict):
        if self.trainer.is_master or not self.master_only:
            self._before_step(feed_dict)

    def _before_step(self, feed_dict):
        """
        Called before every step.
        """
        pass

    def after_step(self, feed_dict):
        if self.trainer.is_master or not self.master_only:
            self._after_step(feed_dict)

    def _after_step(self, feed_dict):
        """
        Called after every step.
        """
        pass

    def trigger_step(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger_step()

    def _trigger_step(self):
        """
        Called after after step.
        """
        pass

    def after_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._after_epoch()

    def _after_epoch(self):
        """
        Called after every epoch.
        """
        pass

    def trigger_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after after epoch.
        """
        pass

    def trigger(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this method
        both inside an epoch and after an epoch.
        """
        pass

    def after_train(self):
        if self.trainer.is_master or not self.master_only:
            self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    def save_checkpoint(self, save_dir):
        save_dir = osp.normpath(save_dir)
        if self.trainer.is_master or not self.master_only:
            self._save_checkpoint(save_dir)

    def _save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, load_dir):
        load_dir = osp.normpath(load_dir)
        if self.trainer.is_master or not self.master_only:
            self._load_checkpoint(load_dir)

    def _load_checkpoint(self, load_dir):
        pass

    def __str__(self):
        return type(self).__name__


class LambdaCallback(Callback):
    """
    A callback created with lambda functions.
    """
    def __init__(self,
                 before_train_fn=None,
                 before_epoch_fn=None,
                 before_step_fn=None,
                 after_step_fn=None,
                 trigger_step_fn=None,
                 after_epoch_fn=None,
                 trigger_epoch_fn=None,
                 trigger_fn=None,
                 after_train_fn=None,
                 save_checkpoint_fn=None,
                 load_checkpoint_fn=None,
                 master_only=False):
        self.before_train_fn = before_train_fn
        self.before_epoch_fn = before_epoch_fn
        self.before_step_fn = before_step_fn
        self.after_step_fn = after_step_fn
        self.trigger_step_fn = trigger_step_fn
        self.after_epoch_fn = after_epoch_fn
        self.trigger_epoch_fn = trigger_epoch_fn
        self.trigger_fn = trigger_fn
        self.after_train_fn = after_train_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.load_checkpoint_fn = load_checkpoint_fn
        self.master_only = master_only

    def _before_train(self):
        if self.before_train_fn:
            self.before_train_fn(self)

    def _before_epoch(self):
        if self.before_epoch_fn:
            self.before_epoch_fn(self)

    def _before_step(self, feed_dict):
        if self.before_step_fn:
            self.before_step_fn(self, feed_dict)

    def _after_step(self, feed_dict):
        if self.after_step_fn:
            self.after_step_fn(self, feed_dict)

    def _trigger_step(self):
        if self.trigger_step_fn:
            self.trigger_step_fn(self)

    def _after_epoch(self):
        if self.after_epoch_fn:
            self.after_epoch_fn(self)

    def _trigger_epoch(self):
        if self.trigger_epoch_fn:
            self.trigger_epoch_fn(self)

    def _trigger(self):
        if self.trigger_fn:
            self.trigger_fn(self)

    def _after_train(self):
        if self.after_train_fn:
            self.after_train_fn(self)

    def _save_checkpoint(self, save_dir):
        if self.save_checkpoint_fn:
            self.save_checkpoint_fn(self, save_dir)

    def _load_checkpoint(self, load_dir):
        if self.load_checkpoint_fn:
            self.load_checkpoint_fn(self, load_dir)


class ProxyCallback(Callback):
    """
    A callback which proxy all methods to another callback.
    """
    def __init__(self, callback):
        assert isinstance(callback, Callback), type(callback)
        self.callback = callback

    def _set_trainer(self, trainer):
        self.callback.set_trainer(trainer)

    def _before_train(self):
        self.callback.before_train()

    def _before_epoch(self):
        self.callback.before_epoch()

    def _before_step(self, feed_dict):
        self.callback.before_step(feed_dict)

    def _after_step(self, feed_dict):
        self.callback.after_step(feed_dict)

    def _trigger_step(self):
        self.callback.trigger_step()

    def _after_epoch(self):
        self.callback.after_epoch()

    def _trigger_epoch(self):
        self.callback.trigger_epoch()

    def _trigger(self):
        self.callback.trigger()

    def _after_train(self):
        self.callback.after_train()

    def _save_checkpoint(self, save_dir):
        self.callback.save_checkpoint(save_dir)

    def _load_checkpoint(self, load_dir):
        self.callback.load_checkpoint(load_dir)

    def __str__(self):
        return 'Proxy-' + str(self.callback)


class Callbacks(Callback):
    """
    A container to hold callbacks.
    """
    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = callbacks

    def _set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def _before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def _before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def _before_step(self, feed_dict):
        for callback in self.callbacks:
            callback.before_step(feed_dict)

    def _after_step(self, feed_dict):
        for callback in self.callbacks:
            callback.after_step(feed_dict)

    def _trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def _after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def _trigger_epoch(self):
        for callback in self.callbacks:
            callback.trigger_epoch()

    def _trigger(self):
        for callback in self.callbacks:
            callback.trigger()

    def _after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def _save_checkpoint(self, save_dir):
        for callback in self.callbacks:
            callback.save_checkpoint(save_dir)

    def _load_checkpoint(self, load_dir):
        for callback in self.callbacks:
            callback.load_checkpoint(load_dir)

    def append(self, callback):
        self.callbacks.append(callback)

    def extend(self, callbacks):
        self.callbacks.extend(callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]

    def __len__(self):
        return len(self.callbacks)
