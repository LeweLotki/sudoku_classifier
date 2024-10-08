import torch
import torch.nn as nn

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_features, num_classes=5):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        conv_output_size = input_features // 4 

        self.fc1 = nn.Linear(64 * conv_output_size, 128) 
        self.fc2 = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  

        return x

