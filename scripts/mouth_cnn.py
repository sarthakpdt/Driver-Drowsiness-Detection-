#mouth_cnn.py 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MouthCNN(nn.Module):
    def __init__(self):
        super(MouthCNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        x = torch.randn(1,1,24,24)
        x = self.convs(x)   
        self._to_linear = x.numel()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x