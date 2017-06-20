#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:44:55 2017

Analyze the test data to see how a nonlinear fit does at classifying the widths

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
from scipy.optimize import curve_fit

from logistic import load_data


def Gauss(x,*p):
    return p[0]*numpy.exp(-(x-p[1])**2./(2.*p[2]**2.))

def gauss_fit():
    datasets = load_data()
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    x = numpy.arange(-10,10,0.1)
    outWidths = numpy.zeros(len(test_set_x))
    outWidths_err = numpy.zeros(len(test_set_x))
    outWidths_unscl = numpy.zeros(len(test_set_x))
    for i in range(len(test_set_x)):
        popt, pcov = curve_fit(Gauss,x,test_set_x[i],p0=(1.0,0.,1.))
        outWidths_unscl[i] = popt[2]
        outWidths[i] = int(round(popt[2],1)*10)
        outWidths_err[i] = numpy.sqrt(pcov[2][2])
    return test_set_y.eval(), outWidths, outWidths_err, outWidths_unscl
    
if __name__ == '__main__':
    true_y_vals, pred_y_vals, pred_y_var, pred_y_unscl = gauss_fit()
    wrong = numpy.array([])
    mis_true_y = numpy.array([])
    mis_pred_y = numpy.array([])
    counter = 0
    counter1 = 0
    for i in range(len(true_y_vals)):
        if numpy.abs(pred_y_unscl[i]*10.-true_y_vals[i]) >= 2.*pred_y_var[i]:
            counter1 += 1
        if true_y_vals[i] != pred_y_vals[i]:
            counter += 1
            wrong = numpy.append(wrong,i)
            mis_true_y = numpy.append(mis_true_y,true_y_vals[i])
            mis_pred_y = numpy.append(mis_pred_y,pred_y_vals[i])
    print counter/20000.
    print counter1/20000.
    print mis_true_y
    print mis_pred_y
        