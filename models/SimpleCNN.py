import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1   = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2   = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(400, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

