import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    


    def __init__(self,num_classes,filename):
        super(Net, self).__init__()
        self.filename = filename
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(43264, num_classes*2)
        self.fc2 = nn.Linear(num_classes*2, num_classes)
        self.drop = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x

    def load(self):
        self.load_state_dict(torch.load(self.filename))

    def save(self):
        torch.save(self.state_dict(), self.filename)
