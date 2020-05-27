import multiprocessing as mp
import os
import time
from queue import Empty

import numpy as np
import torch
from tensorpack.utils.concurrency import (ensure_proc_terminate,
                                          start_proc_mask_signal)
from tensorpack.utils.nvml import NVMLContext

from ..logging import logger
from .callback import Callback

__all__ = ['GPUUtilizationTracker', 'ThroughputTracker']


class GPUUtilizationTracker(Callback):
    """
    Track the average GPU utilization within an epoch.
    It will start a process to track GPU utilization through NVML
    every second within the epoch (the time of `trigger_epoch` is not included).
    This callback creates a process, therefore it is not safe to be used with MPI.
    """
    master_only = True

    def __init__(self, *, devices=None):
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
        except:
            queue.put(None)

    def _before_train(self):
        self.queue = mp.Queue()
        self.event = mp.Event()
        self.process = mp.Process(target=self._worker,
                                  args=(self.devices, self.queue, self.event))
        ensure_proc_terminate(self.process)
        start_proc_mask_signal(self.process)

    def _before_epoch(self):
        while self.event.is_set():
            pass
        self.event.set()

    def _after_epoch(self):
        while self.event.is_set():
            pass
        self.event.set()

    def _trigger_epoch(self):
        try:
            meters = self.queue.get(timeout=60)
        except Empty:
            meters = None

        if meters is None:
            logger.exception('Error in `GPUUtilizationTracker` worker.')
            return

        self.trainer.monitors.add_scalar('utilization/gpu', np.mean(meters))
        if len(self.devices) > 1:
            for k, device in enumerate(self.devices):
                self.trainer.monitors.add_scalar(
                    'utilization/gpu{}'.format(device), meters[k])

    def _after_train(self):
        if self.process.is_alive():
            self.process.terminate()


class ThroughputTracker(Callback):
    """
    Track the throughput within an epoch (the time of `trigger_epoch` is not included).
    """
    master_only = True

    def __init__(self, *, samples_per_step=None):
        self.samples_per_step = samples_per_step

    def _before_train(self):
        self.last_step = self.trainer.global_step

    def _before_epoch(self):
        self.start_time = time.time()

    def _after_epoch(self):
        self.end_time = time.time()

    def _trigger_epoch(self):
        steps_per_sec = (self.trainer.global_step -
                         self.last_step) / (self.end_time - self.start_time)
        self.last_step = self.trainer.global_step

        if self.samples_per_step is None:
            self.trainer.monitors.add_scalar('throughput/steps_per_sec',
                                             steps_per_sec)
        else:
            samples_per_sec = steps_per_sec * self.samples_per_step
            self.trainer.monitors.add_scalar('throughput/samples_per_sec',
                                             samples_per_sec)
