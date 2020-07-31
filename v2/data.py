# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:46:34 2020

@author: xw
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re

# Loading data files

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

def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def prepareData(name, max_length):
    inputs_dict, target_dict, pairs = ReadData(name)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        inputs_dict.addSentence(pair[0])
        target_dict.addWord(pair[1])
    print("Counted words:")
    print(inputs_dict.name, inputs_dict.n_words)
    print(target_dict.name, target_dict.n_words)
    return inputs_dict, target_dict, pairs