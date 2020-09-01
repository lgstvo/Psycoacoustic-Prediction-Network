import torch
import numpy as np
import torch.nn as nn

class BaseCNN(nn.Model):

    def __init__(self):
        super(Net, self).__init__()

        # Input Layer
        self.conv1 = nn.Conv1d(16000, 512, stride = 10)
        self.bnorm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLu()
        self.maxpool1 = nn.MaxPool1d(2, stride = 2)

        # Convolutional Unit 1
        self.conv2 = nn.Conv1d(512, 256, stride = 5)
        self.bnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLu()
        self.maxpool2 = nn.MaxPool1d(2, stride = 2)

        # Convolutional Unit 2
        self.conv3 = nn.Conv1d(256, 128, stride = 2)
        self.bnorm3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLu()
        self.maxpool3 = nn.MaxPool1d(2, stride = 2)

        # Convolutional Unit 3
        self.conv4 = nn.Conv1d(128, 64, stride = 2)
        self.bnorm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLu()
        self.maxpool4 = nn.MaxPool1d(2, stride = 2)

        # Convolutional Unit 4
        self.conv5 = nn.Conv1d(64, 32, stride = 1)
        self.bnorm5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLu()
        self.maxpool5 = nn.MaxPool1d(2, stride = 2)

        # Dropout
        self.drop1 = nn.Dropout(p=0.3)

        # Fully Connected 1
        self.fc1 = nn.Linear(32, 5)

        # Fully Connected 2
        self.fc2 = nn.Linear(5, 1)

    def foward(self, x):

        out = self.conv1(x)
        out = self.bnorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.bnorm3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.conv4(out)
        out = self.bnorm4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        out = self.conv5(out)
        out = self.bnorm5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out