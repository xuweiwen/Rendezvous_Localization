# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:53:31 2020

@author: xw
"""
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class Context(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Context, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def forward(self, input, hidden):    
        embedded = self.embedding(input).view(1, 1, -1)
        _, hidden = self.lstm(embedded, hidden)
        return hidden

class Classification(nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 1)
        self.pool = nn.MaxPool2d((1, 2), 2)
        self.conv2 = nn.Conv2d(4, 4, 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        #x = x.view(1, 1, math.sqrt(x.size()[3]), math.sqrt(x.size()[3]))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sm(x)
        return x