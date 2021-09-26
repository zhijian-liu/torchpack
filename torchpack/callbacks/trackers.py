import multiprocessing as mp
import os
import time
from queue import Empty, Queue
from typing import List, Optional

import numpy as np
import torch.cuda
from tensorpack.utils.concurrency import (ensure_proc_terminate,
                                          start_proc_mask_signal)
from tensorpack.utils.nvml import NVMLContext

from torchpack.utils.logging import logger

from .callback import Callback

__all__ = ['GPUUtilizationTracker', 'ThroughputTracker']


class GPUUtilizationTracker(Callback):
    """Track the average GPU utilization within an epoch.

    It will start a process to track GPU utilization through NVML every second
    within the epoch (the time of `trigger_epoch` is not included). This
    callback creates a process, therefore it is not safe to be used with MPI.
    """

    master_only: bool = True

    def __init__(self, *, devices: Optional[List[int]] = None) -> None:
        if devices is not None:
            self.devices = devices
        else:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if env:
                self.devices = list(map(int, env.split(',')))
            elif env is None:
                self.devices = list(range(torch.cuda.device_count()))
                if len(self.devices) > 1:
                    logger.warning(
                        'Neither `devices` nor `CUDA_VISIBLE_DEVICES` is set! '
                        'All {} visible GPUs will be monitored.'.format(
                            len(self.devices)))
            else:
                raise RuntimeError('No GPU device is specified!')

    @staticmethod
    def _worker(devices, queue, event):
        try:
            with NVMLContext() as ctx:
                while True:
                    event.wait()
                    event.clear()
                    meters = []
                    while not event.is_set():
                        time.sleep(1)
                        meters.append([
                            ctx.device(k).utilization()['gpu'] for k in devices
                        ])
                    meters = meters[:max(len(meters) - 1, 1)]
                    queue.put(np.mean(meters, axis=0))
                    event.clear()
        except Exception:
            queue.put(None)

    def _before_train(self) -> None:
        self.queue: Queue[np.ndarray] = mp.Queue()
        self.event = mp.Event()
        self.process = mp.Process(
            target=self._worker,
            args=(self.devices, self.queue, self.event),
        )
        ensure_proc_terminate(self.process)
        start_proc_mask_signal(self.process)

    def _before_epoch(self) -> None:
        while self.event.is_set():
            pass
        self.event.set()

    def _after_epoch(self) -> None:
        while self.event.is_set():
            pass
        self.event.set()

    def _trigger_epoch(self) -> None:
        try:
            meters = self.queue.get(timeout=60)
        except Empty:
            logger.exception('Error occurred in `GPUUtilizationTracker`.')
            return

        self.trainer.summary.add_scalar('utilization/gpu', np.mean(meters))
        if len(self.devices) > 1:
            for k, device in enumerate(self.devices):
                self.trainer.summary.add_scalar(
                    f'utilization/gpu{device}',
                    meters[k],
                )

    def _after_train(self) -> None:
        if self.process.is_alive():
            self.process.terminate()


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
