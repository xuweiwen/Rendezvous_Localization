# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 04:34:37 2020

@author: xw
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, embedding_dim):
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.embedding_dim = embedding_dim
    
    self.embedding = nn.Embedding(input_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_size)
    
        
  def init_hidden(self):
    return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
  
  def forward(self, input, hidden):
    
    embedded = self.embedding(input)
    output, hidden = self.lstm(embedded, hidden)
    return output, hidden

class AttentionDecoder(nn.Module):
  
  def __init__(self, hidden_size, output_size, embedding_dim, dropout_p = 0.1, max_length = MAX_LENGTH):
    super(AttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.embedding_dim = embedding_dim
    self.max_length = max_length
    
    self.embedding = nn.Embedding(output_size, embedding_dim)
    self.attn = nn.Linear(hidden_size + output_size, max_length)
    self.attn_combine = nn.Linear(hidden_size + output_size, hidden_size)
    self.dropout = nn.Dropout(dropout_p)
    self.lstm = nn.LSTM(embedding_dim, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
  
  def init_hidden(self):
    return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
  
  def forward(self, input, decoder_hidden, encoder_outputs):
    
    embedded = self.embedding(input)
    embedded = self.dropout(embedded)
    attn_weights = F.softmax(self.attn(torch.cat((embedded[0], decoder_hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
    
    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    
    output = F.relu(output)
    output, decoder_hidden = self.lstm(output, decoder_hidden)
    
    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, decoder_hidden, attn_weights