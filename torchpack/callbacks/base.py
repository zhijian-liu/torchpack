from abc import ABCMeta

import six

__all__ = ['Callback', 'ProxyCallback']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks. See
    `Write a Callback
    <http://tensorpack.readthedocs.io/tutorial/extend/callback.html>`_
    for more detailed explanation of the callback methods.

    Attributes:
        epoch_num(int): trainer.epoch_num
        global_step(int): trainer.global_step
        local_step(int): trainer.local_step
        trainer(Trainer): the trainer.
        graph(tf.Graph): the graph.

    Note:
        These attributes are available only after (and including)
        :meth:`_setup_graph`.

    .. document private functions
    .. automethod:: _setup_graph
    .. automethod:: _before_train
    .. automethod:: _after_train
    .. automethod:: _before_run
    .. automethod:: _after_run
    .. automethod:: _before_epoch
    .. automethod:: _after_epoch
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    .. automethod:: _trigger
    """

    def __init__(self):
        self.trainer = None

    def before_train(self):
        self._before_train()

    def _before_train(self):
        """
        Called right before the first iteration. The main difference to
        `setup_graph` is that at this point the graph is finalized and a default session is initialized.
        Override this method to, e.g. run some operations under the session.

        This is similar to ``tf.train.SessionRunHook.after_create_session()``, but different:
        it is called after the session is initialized by :class:`tfutils.SessionInit`.
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

    def before_run(self):
        self._before_run()

    def _before_run(self):
        pass

    def after_run(self):
        self._after_run()

    def _after_run(self):
        """
        It is called after every ``hooked_sess.run()`` call, and it
        processes the values requested by the corresponding :meth:`before_run`.
        It is equivalent to ``tf.train.SessionRunHook.after_run()``, refer to
        TensorFlow docs for more details.
        """
        pass

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        """
        Called after each :meth:`Trainer.run_step()` completes. Defaults to no-op.

        You can override it to implement, e.g. a ProgressBar.
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
    def epoch_num(self):
        return self.trainer.epoch_num

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def local_step(self):
        return self.trainer.local_step

    def __str__(self):
        return type(self).__name__


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, callback):
        """
        Args:
            callback(Callback): the underlying callback
        """
        super(ProxyCallback, self).__init__()
        assert isinstance(callback, Callback), type(callback)
        self.callback = callback

    def _before_train(self):
        self.callback.before_train()

    def _trigger_epoch(self):
        self.callback.trigger_epoch()

    def _trigger(self):
        self.callback.trigger()

    def _trigger_step(self):
        self.callback.trigger_step()

    def _after_train(self):
        self.callback.after_train()

    def _before_epoch(self):
        self.callback.before_epoch()

    def _after_epoch(self):
        self.callback.after_epoch()

    def _before_run(self):
        self.callback._before_run()

    def _after_run(self):
        self.callback._after_run()

    def __str__(self):
        return 'Proxy-' + str(self.callback)
