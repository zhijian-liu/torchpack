import multiprocessing as mp
import os
import time

import numpy as np
import torch
from six.moves import map, queue
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.nvml import NVMLContext
from tensorpack.utils.timer import Timer

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['GPUUtilizationTracker', 'ThroughputTracker']


class GPUUtilizationTracker(Callback):
    """
    Summarize the average GPU utilization within an epoch.
    It will start a process to obtain GPU utilization through NVML every second
    within the epoch (the trigger_epoch time was not included),
    and write average utilization to monitors.
    This callback creates a process, therefore it's not safe to be used with MPI.
    """

    chief_only = False

    def __init__(self, devices=None):
        """
        Args:
            devices (list[int]): physical GPU ids. If None, will use CUDA_VISIBLE_DEVICES
        """
        if devices is None:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if env is None:
                self.devices = list(range(torch.cuda.device_count()))
                if len(self.devices) > 1:
                    logger.warning('Both devices and CUDA_VISIBLE_DEVICES are None! '
                                   'Will monitor all {} visible GPUs!'.format(len(self.devices)))
            elif env:
                self.devices = list(map(int, env.split(',')))
            else:
                self.devices = []
        else:
            self.devices = devices
        assert len(self.devices), 'No GPU device is given!'

    def _before_train(self):
        self.event = mp.Event()
        self.stop_event = mp.Event()
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.worker, args=(
            self.event, self.queue, self.stop_event, self.devices))
        ensure_proc_terminate(self.process)
        start_proc_mask_signal(self.process)

    def _before_epoch(self):
        self.event.set()

    def _after_epoch(self):
        while self.event.is_set():  # unlikely, unless the epoch is extremely fast
            pass
        self.event.set()

    def _trigger_epoch(self):
        try:
            results = self.queue.get(timeout=60)
        except queue.Empty:
            if self.process.is_alive():
                raise RuntimeError("GPUUtilization.worker() is stuck. This is a bug.")
            else:
                raise RuntimeError("GPUUtilization.worker() process is killed unexpectedly.")

        if isinstance(results, int) and results == -1:
            from torchpack.train.exception import StopTraining
            raise StopTraining("GPUUtilizationTracker.worker has failed.")

        self.trainer.monitors.add_scalar('utilization/gpu', np.mean(results))
        for k, device in enumerate(self.devices):
            self.trainer.monitors.add_scalar('utilization/gpu{}'.format(device), results[k])

    def _after_train(self):
        self.stop_event.set()
        self.event.set()
        self.process.terminate()

    @staticmethod
    def worker(event, queue, stop_event, devices):
        """
        Args:
            devices (list[int])
        """
        with NVMLContext() as ctx:
            devices = [ctx.device(i) for i in devices]
            while True:
                try:
                    event.wait()  # start epoch
                    event.clear()
                    if stop_event.is_set():  # or on exit
                        return

                    stats = np.zeros((len(devices),), dtype='f4')
                    cnt = 0
                    while True:
                        time.sleep(1)

                        data = [d.utilization()['gpu'] for d in devices]
                        data = list(map(float, data))
                        stats += data
                        cnt += 1

                        if event.is_set():  # stop epoch
                            if stop_event.is_set():  # or on exit
                                return
                            event.clear()
                            if cnt > 1:
                                # Ignore the last datapoint. Usually is zero, makes us underestimate the util.
                                stats -= data
                                cnt -= 1
                            queue.put(stats / cnt)
                            break
                except Exception:
                    logger.exception("Exception in GPUUtilizationTracker.worker")
                    queue.put(-1)
                    return


class ThroughputTracker(Callback):
    """
    This callback writes the training throughput (in terms of either steps/sec, or samples/sec)
    to the monitors everytime it is triggered.
    The throughput is computed based on the duration between the consecutive triggers.
    The time spent on callbacks after each epoch is excluded.
    """

    chief_only = False

    def __init__(self, samples_per_step=None):
        """
        Args:
            samples_per_step (int or None): total number of samples processed in each step
                (i.e., your total batch size in each step).
                If not provided, this callback will record "steps/sec" instead of "samples/sec".
        """
        self.samples_per_step = samples_per_step
        self.timer = Timer()
        self.timer.pause()

    # only include the time between before_epoch/after_epoch
    def _before_epoch(self):
        self.timer.resume()

    def _after_epoch(self):
        self.timer.pause()

    def _before_train(self):
        self._update_last()

    def _update_last(self):
        old_pause = self.timer.is_paused()
        self.timer.reset()
        if old_pause:
            self.timer.pause()
        self.last_step = self.trainer.global_step

    def _trigger_epoch(self):
        steps_per_sec = (self.trainer.global_step - self.last_step) / self.timer.seconds()
        self._update_last()

        if self.samples_per_step is None:
            self.trainer.monitors.add_scalar('throughput', steps_per_sec)
        else:
            self.trainer.monitors.add_scalar("Throughput (samples/sec)", steps_per_sec * self.samples_per_step)
