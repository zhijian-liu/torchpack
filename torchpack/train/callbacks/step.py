from torchpack.utils.logging import logger

from .callback import Callback

__all__ = ['MaintainStepCounter']


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the trainer, making sure it's increased by one at every `hooked_sess.run`.
    This callback is used internally by the trainer, you don't need to worry about it.
    """

    _chief_only = False

    # def _setup_graph(self):
    #     # ensure it exists
    #     gs_var = get_global_step_var()
    #     with tf.name_scope(None):
    #         self.gs_incr_op = tf.assign_add(
    #             gs_var, 1,
    #             name=GLOBAL_STEP_INCR_OP_NAME).op
    #     self._fetches = tf.train.SessionRunArgs(self.gs_incr_op)

    def _before_train(self):
        if self.trainer.global_step != 0:
            logger.info("Start training with global_step={}".format(self.trainer.global_step))

    def _trigger_step(self):
        self.trainer.loop._global_step += 1
