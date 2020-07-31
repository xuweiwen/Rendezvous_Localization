# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:56:50 2020

@author: xw
"""
from __future__ import unicode_literals, print_function, division
import random
import math
import time

import torch
import torch.nn as nn
from torch import optim

# Preparing Training Data

def indexesFromSentence(dict, sentence):
    return [dict.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(dict, sentence, device):
    indexes = indexesFromSentence(dict, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, device, inputs_dict, target_dict):
    inputs_tensor = tensorFromSentence(inputs_dict, pair[0], device)
    target_tensor = torch.zeros(1, dtype=torch.long)
    target_tensor[0] = target_dict.word2index[pair[1]]
    return (inputs_tensor, target_tensor)

# Training the Model

def train(inputs_tensor, target_tensor, context, classification, context_optimizer, classification_optimizer, criterion):
    
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

def trainIters(context, classification, device, inputs_dict, target_dict, pairs, n_iters, print_every=1000, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    context_optimizer = optim.SGD(context.parameters(), lr=learning_rate)
    classification_optimizer = optim.SGD(classification.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), device, inputs_dict, target_dict)
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        inputs_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(inputs_tensor, target_tensor, context,
                     classification, context_optimizer, classification_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    #print(plot_losses)
    #showPlot(plot_losses)
    return context, classification, plot_losses