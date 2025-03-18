#*********************************
# Fully-connected layer
# Author: Manuel Serna-Aguilera
#*********************************

import activations
import numpy as np



class FullyConnected():
    #=============================
    # Init
    '''
    Input:
        f: activation function for this layer
            Possible values for f as of now:
                'sigmoid'
                'tanh'
                'softmax'
    Return: NA
    '''
    #=============================
    def __init__(self, f=''):
        # Forward computations; these will be stored for backwards computations
        self.x = None # layer input signal
        self.z = None # weighted sum (forward)
        self.a = None # activation (forward)
        self.f = None
        
        # Set activation
        if f == 'sigmoid':
            self.f = activations.Sigmoid()
        elif f == 'tanh':
            self.f = activations.Tanh()
        elif f == 'softmax':
            self.f = activations.Softmax()
        else:
            raise Exception('Currently need to specify activation.')
        
        # Back computations
        self.delta = None # delta for this layer
    
    #=============================
    # Compute forward step for one layer
    '''
    Input:
        x: layer input
        W: layer weight
        b: layer bias
    Return: NA
    '''
    #=============================
    def forward(self, x, W, b) -> None:
        self.x = x
        self.z = (W @ x) + b
        self.a = self.f.forward(self.z)
    
    #=============================
    # Backpropagate through one layer
    '''
    Input:
        next_delta: delta from layer l+1
        W:          weights for current layer l
        b:          biases for current layer l
    Return:
        dWl:        gradients for W for current layer l
        dbl:        gradients for b for current layer l
        delta:      delta for current layer l
    '''
    #=============================
    def backward(self, next_delta, W, b) -> tuple:
        delta = self.f.backward(self.z) * (W.T @ next_delta)
        dWl = delta @ self.x.T
        dbl = delta
        
        return dWl, dbl, delta
    
