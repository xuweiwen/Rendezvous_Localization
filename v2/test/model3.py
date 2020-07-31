# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 04:34:37 2020

@author: Wei Xu
"""

#%% Requirements

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Loading data files

class Dict:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s

def ReadData(name):
    print("Reading lines...")
    lines = open('%s.txt' %name, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    inputs_dict = Dict(name)
    target_dict = Dict('%s_target' %name)
    return inputs_dict, target_dict, pairs

MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(name):
    inputs_dict, target_dict, pairs = ReadData(name)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        inputs_dict.addSentence(pair[0])
        target_dict.addWord(pair[1])
    print("Counted words:")
    print(inputs_dict.name, inputs_dict.n_words)
    print(target_dict.name, target_dict.n_words)
    return inputs_dict, target_dict, pairs

inputs_dict, target_dict, pairs = prepareData('name')
print(random.choice(pairs))

#%% Model

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

#%% Training

# Preparing Training Data

def indexesFromSentence(dict, sentence):
    return [dict.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(dict, sentence):
    indexes = indexesFromSentence(dict, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    inputs_tensor = tensorFromSentence(inputs_dict, pair[0])
    target_tensor = torch.zeros(1, dtype=torch.long)
    target_tensor[0]=target_dict.word2index[pair[1]]
    return (inputs_tensor, target_tensor)

def classFromTensor(result):
    index = torch.max(result, 1)[1]
    return target_dict.index2word[index.item()]

# Training the Model

def train(inputs_tensor, target_tensor, context, classification, context_optimizer, classification_optimizer, criterion, max_length=MAX_LENGTH):
    
    context_hidden = context.init_hidden()

    context_optimizer.zero_grad()
    classification_optimizer.zero_grad()

    inputs_length = inputs_tensor.size(0)

    loss = 0

    for ei in range(inputs_length):
        context_hidden = context(inputs_tensor[ei], context_hidden)
    
    classification_inputs = context_hidden[0]
    
    result = classification(classification_inputs)
    
    loss = criterion(result, target_tensor)
    loss.backward()
    
    context_optimizer.step()
    classification_optimizer.step()
    
    return loss.item()

import time

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(context, classification, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    #plot_losses = []
    print_loss_total = 0  # Reset every print_every
    #plot_loss_total = 0  # Reset every plot_every

    context_optimizer = optim.SGD(context.parameters(), lr=learning_rate)
    classification_optimizer = optim.SGD(classification.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        inputs_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(inputs_tensor, target_tensor, context,
                     classification, context_optimizer, classification_optimizer, criterion)
        print_loss_total += loss
        #plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
'''
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
'''

    #showPlot(plot_losses)
'''
# Plotting results

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
'''
# Evaluation

def evaluate(context, classification, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(inputs_dict, sentence)
        input_length = input_tensor.size()[0]
        context_hidden = context.init_hidden()

        for ei in range(input_length):
            context_hidden = context(input_tensor[ei], context_hidden)

        classification_inputs = context_hidden[0]
        
        result = classification(classification_inputs)
        
        result = classFromTensor(result)
        
        return result
        
def evaluateRandomly(context, classification, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        result = evaluate(context, classification, pair[0])
        print('<', result)
        print('')

# Training and Evaluating

hidden_size = 256
context_x = Context(inputs_dict.n_words, hidden_size).to(device)
classification_x = Classification().to(device)

trainIters(context_x, classification_x, 75000, print_every=5000)

evaluateRandomly(context_x, classification_x)