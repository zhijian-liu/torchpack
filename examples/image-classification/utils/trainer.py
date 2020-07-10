from torchpack.engine import Trainer

__all__ = ['ClassificationTrainer']


class ClassificationTrainer(Trainer):
    def __init__(self, *, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _before_epoch(self):
        self.model.train()

    def _run_step(self, feed_dict):
        inputs = feed_dict['image'].cuda(non_blocking=True)
        targets = feed_dict['class'].cuda(non_blocking=True)

        outputs = self.model(inputs)

        if self.model.training:
            loss = self.criterion(outputs, targets)
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self):
        self.model.eval()
        self.scheduler.step()

    def _state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def _load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
