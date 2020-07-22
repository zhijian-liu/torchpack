from typing import Any, Dict

from torchpack.train import Trainer

__all__ = ['ClassificationTrainer']


class ClassificationTrainer(Trainer):
    def __init__(self, *, model, criterion, optimizer, scheduler) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs = feed_dict['image'].cuda(non_blocking=True)
        targets = feed_dict['class'].cuda(non_blocking=True)

        outputs = self.model(inputs)

        if outputs.requires_grad:
            loss = self.criterion(outputs, targets)
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict.pop('model'))
        self.optimizer.load_state_dict(state_dict.pop('optimizer'))
        self.scheduler.load_state_dict(state_dict.pop('scheduler'))
