from typing import Any, Callable, Dict

from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['ClassificationTrainer']


class ClassificationTrainer(Trainer):

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        scheduler: Scheduler,
        amp_enabled: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs = feed_dict['image'].cuda(non_blocking=True)
        targets = feed_dict['class'].cuda(non_blocking=True)

        with amp.autocast(enabled=self.model.training and self.amp_enabled):
            outputs = self.model(inputs)

            if self.model.training:
                loss = self.criterion(outputs, targets)
                self.summary.add_scalar('loss', loss.item())

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict.pop('model'))
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict.pop('optimizer'))
        self.scheduler.load_state_dict(state_dict.pop('scheduler'))
