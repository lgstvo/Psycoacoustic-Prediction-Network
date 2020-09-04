import torch
import numpy as np
import torch.nn as nn

class BaseCNN(nn.Module):

    def __init__(self):
        super(BaseCNN, self).__init__()

        # [batch_size, channel_size, time_stamp] = [1, 1, 16000]
        # Input Layer
        self.conv1 = nn.Conv1d(1, 10, kernel_size=(512), stride=10)
        self.bnorm1 = nn.BatchNorm1d(10)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d((2), stride=2)

        # Convolutional Unit 1
        self.conv2 = nn.Conv1d(10, 20, kernel_size=(256), stride=5)
        self.bnorm2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d((2), stride=2)

        # Convolutional Unit 2
        self.conv3 = nn.Conv1d(20, 40, kernel_size=(128), stride=2)
        self.bnorm3 = nn.BatchNorm1d(40)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d((2), stride=2)

        # Convolutional Unit 3
        self.conv4 = nn.Conv1d(40, 60, kernel_size=(64), stride=2)
        self.bnorm4 = nn.BatchNorm1d(60)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d((2), stride=2)

        # Convolutional Unit 4
        self.conv5 = nn.Conv1d(60, 80, kernel_size=(32), stride=1)
        self.bnorm5 = nn.BatchNorm1d(80)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d((2), stride=2)

        # Dropout
        self.drop1 = nn.Dropout(p=0.3)

        # Fully Connected 1
        self.fc1 = nn.Linear(32, 5)

        # Fully Connected 2
        self.fc2 = nn.Linear(5, 1)

    def foward(self, x): 

        print("Input")
        print(x.size())
        print("Camada [1]: conv1")
        out = self.conv1(x)
        print(out.size())
        print("Camada [2]: bnorm1")
        out = self.bnorm1(out)
        print(out.size())
        print("Camada [3]: relu1")
        out = self.relu1(out)
        print(out.size())
        print("Camada [4]: maxpool1")
        out = self.maxpool1(out)
        print(out.size())

        print("Camada [6]: conv2")
        out = self.conv2(out)
        print(out.size())
        print("Camada [7]")
        out = self.bnorm2(out)
        print(out.size())
        print("Camada [8]")
        out = self.relu2(out)
        print(out.size())
        print("Camada [9]")
        out = self.maxpool2(out)
        print(out.size())

        out = self.conv3(out)
        out = self.bnorm3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        print(out.size())

        out = self.conv4(out)
        out = self.bnorm4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        print(out.size())

        out = self.conv5(out)
        out = self.bnorm5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)
        print(out.size())

        out = self.drop1(out)

        out = self.fc1(out)

        out = self.fc2(out)
        
        print(out.size())
        return out