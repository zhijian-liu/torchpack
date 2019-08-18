import multiprocessing as mp
import os
import queue
import time

import numpy as np
import torch
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.nvml import NVMLContext
from tensorpack.utils.timer import Timer

from torchpack.callbacks.callback import Callback
from torchpack.train.exception import StopTraining
from torchpack.utils.logging import logger

__all__ = ['GPUUtilizationTracker', 'ThroughputTracker']


class GPUUtilizationTracker(Callback):
    """
    Track the average GPU utilization within an epoch.
    It will start a process to track GPU utilization through NVML every second
    within the epoch (the `trigger_epoch` time is not included).
    This callback creates a process, therefore it's not safe to be used with MPI.
    """

    master_only = True

    def __init__(self, devices=None):
        """
        Args:
            devices: list of physical GPU IDs. If None, will use CUDA_VISIBLE_DEVICES.
        """
        if devices is not None:
            self.devices = devices
        else:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if env:
                self.devices = list(map(int, env.split(',')))
            elif env is None:
                self.devices = list(range(torch.cuda.device_count()))
                if len(self.devices) > 1:
                    logger.warning('Neither `devices` nor `CUDA_VISIBLE_DEVICES` is set! '
                                   'All {} visible GPUs will be monitored.'.format(len(self.devices)))
            else:
                raise RuntimeError('No GPU device is specified!')

    @staticmethod
    def _worker(devices, event, stop_event, queue):
        with NVMLContext() as ctx:
            devices = [ctx.device(i) for i in devices]
            while True:
                try:
                    event.wait()
                    event.clear()
                    if stop_event.is_set():
                        return

                    count = 0
                    stats = np.zeros((len(devices),), dtype='f4')
                    while True:
                        time.sleep(1)

                        data = [device.utilization()['gpu'] for device in devices]
                        data = list(map(float, data))
                        count += 1
                        stats += data

                        if event.is_set():
                            event.clear()
                            if stop_event.is_set():
                                return
                            if count > 1:
                                count -= 1
                                stats -= data
                            queue.put(stats / count)
                            break
                except Exception:
                    logger.exception('Error occurred in `GPUUtilizationTracker` worker.')
                    queue.put(None)
                    return

    def _before_train(self):
        self.event = mp.Event()
        self.stop_event = mp.Event()
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._worker, args=(self.devices, self.event, self.stop_event, self.queue))
        ensure_proc_terminate(self.process)
        start_proc_mask_signal(self.process)

    def _before_epoch(self):
        self.event.set()

    def _after_epoch(self):
        while self.event.is_set():
            pass
        self.event.set()

    def _trigger_epoch(self):
        try:
            results = self.queue.get(timeout=60)
        except queue.Empty:
            if self.process.is_alive():
                raise RuntimeError('`GPUUtilizationTracker` worker stuck. This is a bug!')
            else:
                raise RuntimeError('`GPUUtilizationTracker` worker is killed unexpectedly!')

        if results is None:
            raise StopTraining('`GPUUtilizationTracker` worker has failed!')

        self.trainer.monitors.add_scalar('utilization/gpu', np.mean(results))
        if len(self.devices) > 1:
            for k, device in enumerate(self.devices):
                self.trainer.monitors.add_scalar('utilization/gpu{}'.format(device), results[k])

    def _after_train(self):
        self.stop_event.set()
        self.event.set()
        self.process.terminate()


class ThroughputTracker(Callback):
    """
    Track the throughput everytime it is triggered.
    The throughput is computed based on the duration between the consecutive triggers.
    The time spent on callbacks after each epoch is excluded.
    """

    master_only = True

    def __init__(self, samples_per_step=None):
        """
        Args:
            samples_per_step: total number of samples processed in each step.
                If not provided, this callback will record "steps/sec" instead of "samples/sec".
        """
        self.samples_per_step = samples_per_step
        self.timer = Timer()
        self.timer.pause()

    def _update_last(self):
        old_pause = self.timer.is_paused()
        self.timer.reset()
        if old_pause:
            self.timer.pause()
        self.last_step = self.trainer.global_step

    def _before_train(self):
        self._update_last()

    def _before_epoch(self):
        self.timer.resume()

    def _after_epoch(self):
        self.timer.pause()

    def _trigger_epoch(self):
        steps_per_sec = (self.trainer.global_step - self.last_step) / self.timer.seconds()
        self._update_last()

        if self.samples_per_step is None:
            self.trainer.monitors.add_scalar('throughput', steps_per_sec)
        else:
            self.trainer.monitors.add_scalar('throughput', steps_per_sec * self.samples_per_step)
