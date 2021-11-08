import time
from typing import Optional

from .callback import Callback

__all__ = ['ThroughputTracker']


class ThroughputTracker(Callback):
    """Track the throughput within an epoch.

    Note that the time of `trigger_epoch` is not included.
    """

    master_only: bool = True

    def __init__(self, *, samples_per_step: Optional[int] = None) -> None:
        self.samples_per_step = samples_per_step

    def _before_train(self) -> None:
        self.last_step = self.trainer.global_step

    def _before_epoch(self) -> None:
        self.start_time = time.perf_counter()

    def _after_epoch(self) -> None:
        self.end_time = time.perf_counter()

    def _trigger_epoch(self) -> None:
        steps_per_sec = (self.trainer.global_step
                         - self.last_step) / (self.end_time - self.start_time)
        self.last_step = self.trainer.global_step

        if self.samples_per_step is None:
            self.trainer.summary.add_scalar(
                'throughput/steps_per_sec',
                steps_per_sec,
            )
        else:
            samples_per_sec = steps_per_sec * self.samples_per_step
            self.trainer.summary.add_scalar(
                'throughput/samples_per_sec',
                samples_per_sec,
            )
