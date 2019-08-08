from abc import ABCMeta

import six

__all__ = ['Callback', 'LambdaCallback']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks.

    Attributes:
        trainer(Trainer): the trainer.

    .. document private functions
    .. automethod:: _setup_trainer
    .. automethod:: _before_train
    .. automethod:: _after_train
    .. automethod:: _before_step
    .. automethod:: _after_step
    .. automethod:: _before_epoch
    .. automethod:: _after_epoch
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    """

    _chief_only = True

    def setup_trainer(self, trainer):
        self.trainer = trainer
        self._setup_trainer()

    def _setup_trainer(self):
        """
        Called after finalizing the trainer.
        Override this method to setup the ops used in the callback.
        """
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        """
        Called right before the first iteration. The main difference to
        `setup_graph` is that at this point the graph is finalized and a default session is initialized.
        """
        pass

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        """
        Called right before each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** before each epoch.
        """
        pass

    def after_epoch(self):
        self._after_epoch()

    def _after_epoch(self):
        """
        Called right after each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** after each epoch.
        """
        pass

    def before_step(self, fd):
        self._before_step(fd)

    def _before_step(self, fd):
        """
        It is called before every step, and it registers some extra op/tensors to run in the next call.
        """
        pass

    def after_step(self, fd, od):
        self._after_step(fd, od)

    def _after_step(self, fd, od):
        """
        It is called after every step, and it processes the values requested by the corresponding :meth:`before_run`.
        """
        pass

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        """
        Called after each step completes.
        """
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after the completion of every epoch. Defaults to call ``self.trigger()``
        """
        self.trigger()

    def trigger(self):
        self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this
        method both inside an epoch and after an epoch.

        When used without the scheduler, this method by default will be called by `trigger_epoch()`.
        """
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    @property
    def chief_only(self):
        return self._chief_only

    @chief_only.setter
    def chief_only(self, v):
        self._chief_only = v

    def set_chief_only(self, v=True):
        """
        Set chief_only property, and returns the callback itself.
        """
        self._chief_only = v
        return self

    def __str__(self):
        return type(self).__name__


class LambdaCallback(Callback):
    """
    Create a callback with some lambdas.
    """

    def __init__(self,
                 setup_trainer=None,
                 before_train=None,
                 after_train=None,
                 before_step=None,
                 after_step=None,
                 before_epoch=None,
                 after_epoch=None,
                 trigger_step=None,
                 trigger_epoch=None,
                 trigger=None):
        self._setup_trainer_ = setup_trainer
        self._before_train_ = before_train
        self._after_train_ = after_train
        self._before_step_ = before_step
        self._after_step_ = after_step
        self._before_epoch_ = before_epoch
        self._after_epoch_ = after_epoch
        self._trigger_step_ = trigger_step
        self._trigger_epoch_ = trigger_epoch
        self._trigger_ = trigger

    def _setup_trainer(self):
        if self._setup_trainer_:
            self._setup_trainer_(self)

    def _before_train(self):
        if self._before_train_:
            self._before_train_(self)

    def _after_train(self):
        if self._after_train_:
            self._after_train_(self)

    def _before_step(self, fd):
        if self._before_step_:
            self._before_step_(self, fd)

    def _after_step(self, fd, od):
        if self._after_step_:
            self._after_step_(self, fd)

    def _before_epoch(self):
        if self._before_epoch_:
            self._before_epoch_(self)

    def _after_epoch(self):
        if self._after_epoch_:
            self._after_epoch_(self)

    def _trigger_step(self):
        if self._trigger_step_:
            self._trigger_step_(self)

    def _trigger_epoch(self):
        if self._trigger_epoch_:
            self._trigger_epoch_(self)

    def _trigger(self):
        if self._trigger_:
            self._trigger_(self)
