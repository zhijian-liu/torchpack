from abc import ABCMeta

import six

__all__ = ['Callback']


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
    .. automethod:: _trigger
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

    def before_step(self, ctx):
        fetches = self._before_step(ctx)
        if fetches is None:
            return None

    def _before_step(self, ctx):
        """
        It is called before every step, and it registers some extra op/tensors to run in the next call.
        """
        return None

    def after_step(self, run_context, run_values):
        self._after_step(run_context, run_values)

    def _after_step(self, run_context, run_values):
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
        """
        Only run this callback on chief training process.

        Returns: bool
        """
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
