 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:19:44 2017

This file is used to create a series of Gaussians and their classification
for use as training data in a neural net

For each one-dimensional Gaussian I want to discretize into an array of pixels
and the classification is the width

@author: Chris
"""

import numpy
import random
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

def Gauss(x,*p):
    return p[0]*numpy.exp(-(x-p[1])**2./(2.*p[2]**2.))
x = numpy.arange(-10,10,0.1)
amp = 1.0
noise_mag = amp*0.
fringe_mag = amp*0.4
print(len(x))

dataLen = 100000
counter = 0
dataStor = numpy.zeros((dataLen,len(x)))
widths = numpy.zeros(dataLen)

while counter<dataLen:
    width = round(random.uniform(0.5,1.5),1)
    offset = 0#round(random.uniform(-0.2,0.2),1)
    params = [amp,offset,width]
    y = Gauss(x,*params)
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(y)):
        dataStor[counter][i] = y[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    widths[counter] = width
    counter += 1

tup = (dataStor,widths)      
filehandler = open("TrainData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()

dataLen = 20000
counter = 0
dataStor = numpy.zeros((dataLen,len(x)))
widths = numpy.zeros(dataLen)

while counter<dataLen:
    width = round(random.uniform(0.5,1.5),1)
    offset = 0#round(random.uniform(-0.5,0.5),1)
    params = [amp,offset,width]
    y = Gauss(x,*params)
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(y)):
        dataStor[counter][i] = y[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    widths[counter] = width
    counter += 1

tup = (dataStor,widths)      
filehandler = open("ValidData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()

dataLen = 20000
counter = 0
dataStor = numpy.zeros((dataLen,len(x)))
widths = numpy.zeros(dataLen)

while counter<dataLen:
    width = round(random.uniform(0.5,1.5),1)
    offset = 0#round(random.uniform(-0.5,0.5),1)
    params = [amp,offset,width]
    y = Gauss(x,*params)
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(y)):
        dataStor[counter][i] = y[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    widths[counter] = width
    counter += 1

tup = (dataStor,widths)
plt.plot(x,dataStor[0],'.')
plt.show()      
filehandler = open("TestData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()