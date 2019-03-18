from torchpack.utils.context import *


class ModelDescBase(object):
    def train_step(self):
        raise NotImplementedError()

    def inference_step(self):
        raise NotImplementedError()


class SimpleModelDesc(ModelDescBase):
    _model = None
    _criterion = None
    _optimizer = None

    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @context_outputs('inputs', 'outputs', 'targets', 'loss')
    def train_step(self, data):
        inputs, targets = data
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()
        return outputs, loss

    @context_outputs('inputs', 'outputs', 'targets', 'loss')
    def inference_step(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss
