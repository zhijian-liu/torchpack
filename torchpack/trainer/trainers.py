from .base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, loader, criterion, optimizer):
        super(Trainer, self).__init__()

        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.iterator = iter(self.loader)

    def load_data(self):
        try:
            data = self.iterator.next()
        except:
            self.iterator = iter(self.loader)
            data = self.iterator.next()

        inputs, targets = data
        self.context.update('inputs', inputs)
        self.context.update('targets', targets)

    def run_step(self):
        self.load_data()

        inputs = self.context.get('inputs')
        targets = self.context.get('targets')

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.context.update('outputs', outputs)
        self.context.update('loss', loss)

        self.optimizer.step()
        return outputs, loss
