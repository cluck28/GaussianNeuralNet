#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:37:46 2017

unpack the parameters of the best fit neural network

@author: Chris
"""

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import random

from logistic import load_data

if __name__ == '__main__':
    classifier = pickle.load(open('best_model.pkl'))
    W = classifier.params[0].eval()
    b = classifier.params[1].eval()
    for i in range(len(W)):
        print(W[i])
    for i in range(len(b)):
        print b[i]
    