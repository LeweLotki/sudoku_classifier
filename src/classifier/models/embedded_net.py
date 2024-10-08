import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingConvolutionalNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, num_classes=5):
        super(EmbeddingConvolutionalNetwork, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        conv_output_length = self._compute_conv_output_size(sequence_length)
        conv_output_size = 128 * conv_output_length
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_conv_output_size(self, length):
        length = length
        length = length // 2 
        length = length // 2  
        return length

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

