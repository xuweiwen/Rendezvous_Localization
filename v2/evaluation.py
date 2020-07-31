# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:58:10 2020

@author: xw
"""
from __future__ import unicode_literals, print_function, division
import random
import torch
from train import tensorFromSentence

def classFromTensor(result, target_dict):
    index = torch.max(result, 1)[1]
    return target_dict.index2word[index.item()]

def evaluate(context, classification, device, sentence, inputs_dict, target_dict):
    with torch.no_grad():
        input_tensor = tensorFromSentence(inputs_dict, sentence, device)
        input_length = input_tensor.size()[0]
        context_hidden = context.init_hidden()

        for ei in range(input_length):
            context_hidden = context(input_tensor[ei], context_hidden)

        classification_inputs = context_hidden[0]
        
        result = classification(classification_inputs)
        
        result = classFromTensor(result, target_dict)
        
        return result
        
def evaluateRandomly(context, classification, device, inputs_dict, target_dict, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        result = evaluate(context, classification, device, pair[0], inputs_dict, target_dict)
        print('<', result)
        print('')