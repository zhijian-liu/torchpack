from torchpack.utils.logging import logger

from .base import Callback

__all__ = ['MaintainStepCounter']


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the trainer, making sure it's increased by one at every step.
    This callback is used internally by the trainer, you don't need to worry about it.
    """

    chief_only = False

    def before_train(self):
        if self.trainer.global_step != 0:
            logger.info("Start training with global_step={}".format(self.trainer.global_step))

    def trigger_step(self):
        self.trainer.loop._global_step += 1
