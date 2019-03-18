import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


trainer = None
device = 'cuda'
placeholder = None


class DataDesc():
    @torchpack_outputs('data', ['inputs', 'targets'])
    def run_step(self):
        inputs = 0
        targets = 0


class ModelDesc():
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.optim.SGD()

    @torchpack_inputs('data', ['inputs', 'targets'])
    @torchpack_outputs('model', ['outputs', 'loss'])
    def run_step(self, inputs, targets):
        inputs = inputs.to(device)
        targets = targets.to(device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()

        self.optimizer.step()


class Trainer:
    model = ModelDesc()

    def run_step(self):
        self.model.train_step()
