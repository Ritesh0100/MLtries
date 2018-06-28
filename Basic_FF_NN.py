#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:12:44 2018

@author: ritesh
"""

import numpy as np

#simple feed-forward 3 layer I-H-O NN

#collect data
#x 4x3 matrix
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#y is 4x1 matrix
y = np.array([[0],
              [1],
              [1],
              [0]])


#build model 

num_epochs = 60000 #hyper-parameter

#initialize weights 
#ranges from -1 to 1
#input weights
syn0 = 2*np.random.random((3,4))-1
#output weights
syn1 = 2*np.random.random((4,1))-1


#define sigmoid - non linear function for activation 
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#step 3 train model 
for j in range(num_epochs):
    #feed-forward
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    #how much did we miss the target value?
    l2_error = y - l2
    
    if (j%10000) == 0:
        print("Error:"+str(np.mean(np.abs(l2_error))))
        
    #in what direction is the target value?
    #Error weighted derivative
    l2_delta = l2_error*nonlin(l2,deriv=True)
    
    #how much did each l1 value contribute to l2 error - we are backpropagating
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error*nonlin(l1,deriv = True)
    
    #updaing weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    

#print("True Value:",y)
#print("Prediction:",l2)
    
    
    
