__all__ = ['TrainConfig']


class TrainConfig(object):
    """
    A collection of options to be used for single-cost trainers.

    Note that you do not have to use :class:`TrainConfig`.
    You can use the API of :class:`Trainer` directly, to have more fine-grained control of the training.
    """

    def __init__(self,
                 dataflow=None, data=None,
                 model=None,
                 callbacks=None, extra_callbacks=None, monitors=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999):
        """
        Args:
            dataflow (DataFlow):
            data (InputSource):
            model (ModelDesc):

            callbacks (list[Callback]): a list of :class:`Callback` to use during training.
            extra_callbacks (list[Callback]): This argument
                is only used to provide the defaults in addition to ``callbacks``.
                The list of callbacks that will be used in the end is simply ``callbacks + extra_callbacks``.

                It is usually left as None, and the default value for this argument is :func:`DEFAULT_CALLBACKS()`.
                You can override it when you don't like any of the default callbacks.
                For example, if you'd like to let the progress bar print tensors, you can use

                .. code-block:: none

                    extra_callbacks=[ProgressBar(names=['name']),
                                     MovingAverageSummary(),
                                     MergeAllSummaries(),
                                     RunUpdateOps()]

            monitors (list[MonitorBase]): Defaults to :func:`DEFAULT_MONITORS()`.

            session_init (SessionInit): how to initialize variables of a session. Defaults to do nothing.
            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size. You may want to divide it by the #GPUs in multi-GPU training.
            max_epoch (int): maximum number of epoch to run training.
        """

        # TODO type checker decorator
        def assert_type(v, tp, name):
            assert isinstance(v, tp), \
                "{} has to be type '{}', but an object of type '{}' found.".format(
                    name, tp.__name__, v.__class__.__name__)

        # process data & model
        assert data is None or dataflow is None, "dataflow and data cannot be both presented in TrainConfig!"
        self.dataflow = dataflow
        self.data = data

        # if model is not None:
        #     assert_type(model, ModelDescBase, 'model')
        self.model = model

        # if callbacks is not None:
        #     assert_type(callbacks, list, 'callbacks')
        # self.callbacks = callbacks
        # if extra_callbacks is not None:
        #     assert_type(extra_callbacks, list, 'extra_callbacks')
        # self.extra_callbacks = extra_callbacks
        # if monitors is not None:
        #     assert_type(monitors, list, 'monitors')
        # self.monitors = monitors

        if steps_per_epoch is None:
            steps_per_epoch = len(dataflow)
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
