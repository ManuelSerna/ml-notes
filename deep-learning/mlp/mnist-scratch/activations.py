#*********************************
# Activation Functions Module
# Author: Manuel Serna-Aguilera
#*********************************

import numpy as np



#=================================
# Sigmoid
#=================================
class Sigmoid:
    # Result of sigmoid function
    # Input: z
    def forward(self, z):
        return 1/(1+np.exp(-z))
    
    # Derivative of sigmoid
    # Input: z (same as forward)
    def backward(self, z):
        return self.forward(z) * (1-self.forward(z))

#=================================
# Softmax
#=================================
class Softmax:
    # Make predictions for classes
    # Input: x to make predictions on
    def forward(self, x):
        ex = np.exp(x - np.max(x))
        return (ex/ex.sum())
    
    # Derivative softmax with cross-entropy as loss (simplified)
    # Inputs: x: last layer input; y: true labels
    def backward_crossentropy(self, x, y):
        yhat = self.forward(x) # get prediction vector
        diff = yhat-y
        return diff

#=================================
# Hyperbolic Tangent (tanh)
#=================================
class Tanh:
    # Result of tanh function
    # Input: z: input
    def forward(self, z):
        return np.tanh(z)
    
    # Result of tanh derivative
    # Input: z: input (same as forward)
    def backward(self, z):
        out = self.forward(z)
        dz = 1.0 - np.square(out)
        return dz
