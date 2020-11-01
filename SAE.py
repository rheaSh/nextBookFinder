import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel as parallel
from torch.autograd import Variable


class SAE(nn.Module):
    """
    Implements a Stacked AutoEncoder using PyTorch's Module class.
    Uses linear activation for hidden layers and sigmoid activation for output layer
    Uses 4 fully connected layer with experimental values of number of hidden layers for encoding/decoding
    Doesn't include a training function.
    """

    def __init__(self, n_books=10000):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(n_books, 20)       # Fully Connected Layer1
        self.fc2 = nn.Linear(20, 10)            # Fully Connected Layer2
        self.fc3 = nn.Linear(10, 20)            # Fully Connected Layer3
        self.fc4 = nn.Linear(20, n_books)       # Fully Connected Layer4
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))        # IP x after being encoded once
        x = self.activation(self.fc2(x))        # IP x after being encoded twice
        x = self.activation(self.fc3(x))        # IP x after being decoded once
        x = self.fc4(x)                         # IP x after being decoded twice
        return x
