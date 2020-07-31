# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:59:53 2020

@author: xw
"""

import torch
from data import prepareData
from model import Context, Classification
from train import trainIters
from evaluation import evaluateRandomly
from plot_loss import showPlot

def training():
    context_x = Context(inputs_dict.n_words, hidden_size).to(device)
    classification_x = Classification().to(device)

    context_x, classification_x, plot_losses = trainIters(context_x, classification_x, device, inputs_dict, target_dict, pairs, n_iters, print_every=50)
    
    return context_x, classification_x, plot_losses

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = 'data'
    max_length = 10
    inputs_dict, target_dict, pairs = prepareData(name, max_length)
    #print(target_dict.word2index)
    hidden_size = 256
    n_iters = 700
    context_x, classification_x, plot_losses = training()
    showPlot(plot_losses)
    evaluateRandomly(context_x, classification_x, device, inputs_dict, target_dict, pairs, 5)