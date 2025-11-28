#eye_cnn.py
import torch.nn as nn
import torch.nn.functional as F

class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After 2 poolings:
        # Input: 1x24x24
        # After conv1 -> 16x24x24
        # After pool -> 16x12x12
        # After conv2 -> 32x12x12
        # After pool -> 32x6x6
        # Flatten = 32*6*6 = 1152

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)   

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x