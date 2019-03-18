from .base import Trainer
from ..utils.context import context_save_outputs


class SimpleTrainer(Trainer):
    @property
    def model(self):
        raise NotImplementedError()

    @property
    def criterion(self):
        raise NotImplementedError()

    @property
    def optimizer(self):
        raise NotImplementedError()

    @context_save_outputs('inputs', 'outputs', 'targets', 'loss')
    def run_step(self, data):
        inputs, targets = data
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()
        return outputs, loss
