# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn

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

    def __init__(self, hidden_size, output_size):
        super(Classification, self).__init__()
        self.i2o = nn.Linear(hidden_size, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.i2o(x)
        x = self.sm(x)
        return x