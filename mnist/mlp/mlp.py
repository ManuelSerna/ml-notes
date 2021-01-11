#*********************************
# MLP (Fully-connected) network for MNIST data
# Author: Manuel Serna-Aguilera
#*********************************

import activations
import losses
import mlp_layer
import numpy as np

class MLP():
    #=============================
    # Constructor
    '''
    Input:
        s:         list of width of each layer (including length of input at index 0)
        n_classes: number of outputs/classes
    '''
    #=============================
    def __init__(self, s=None, n_classes=None):
        self.s = s
        self.n_classes = n_classes
        
        # Init parameters (list of np arrays)
        self.W = []
        self.b = []
        
        low = -1.0
        high = 1.0
        
        for i in range(1, len(self.s)):
            w = np.random.uniform(low=low, high=high, size=(s[i], s[i-1]))
            self.W.append(w)
            b = np.zeros((s[i], 1))
            self.b.append(b)
    
    #=============================
    # Forward propagation
    '''
    Input:
        x:      MNIST image (28,28)
    Return:
        layers: list of fully-connected layers (which are objects)
    '''
    #=============================
    def forward(self, x):
        layers = [] # hold layer objects (and store intermediate computations)
        
        # Loop through all layers (weights) except the output layer
        L = len(self.W)
        a = x
        for l in range(L-1):
            layer = mlp_layer.FullyConnected(f='sigmoid')
            layer.forward(a, self.W[l], self.b[l])
            a = layer.a
            layers.append(layer)
        
        # Compute prediction
        layer = mlp_layer.FullyConnected(f='softmax')
        layer.forward(layers[-1].a, self.W[-1], self.b[-1])
        layers.append(layer)
        
        return layers
    
    #=============================
    # Argmax
    # Input: array A
    # Return: index of max element in A
    #=============================
    def get_max_index(self, A):
        max_prob = -1.0
        max_index = -1
        
        for k in range(len(A)):
            if max_prob < A[k]:
                max_prob = A[k]
                max_index = k
        
        return max_index
    
    #=============================
    # Predict
    '''
    Input:
        x: sample input
    Return:
        label: class 
    '''
    #=============================
    def predict(self, x):
        x = x.reshape((len(x)**2, 1)) # reshape 2d data to be (w^2,1)
        
        layers = self.forward(x)
        label = self.get_max_index(layers[-1].a)
        return label
    
    #=============================
    # Evaluate accuracy of model
    '''
    Input:
        X: input data
        Y: corresponding labels
    Return: NA
    '''
    #=============================
    def evaluate(self, X, Y):
        tot_correct = 0
        
        # Iterate over all test samples
        n = len(Y)
        for i in range(n):
            prediction = self.predict(X[i])
            y = Y[i].reshape((len(Y[i]), 1))
            true_label = self.get_max_index(y)
            
            if prediction == true_label:
                tot_correct += 1
        
        print(' Accuracy: {}/{} = {}'.format(tot_correct, n, float(tot_correct/n * 100.0)))
    
    #=============================
    # Backpropagate
    '''
    Input (assumed to be in proper dims so math works out):
        x: reshaped input data
        y: reshaped label
    Return:
        dW, db: gradients of parameters
    '''
    #=============================
    def backprop(self, x, y):
        # Init new gradients (exact same setup as self.W and self.b)
        dW = []
        db = []
        
        for i in range(1, len(self.s)):
            w = np.zeros((self.s[i], self.s[i-1]))
            dW.append(w)
            b = np.zeros((self.s[i], 1))
            db.append(b)
        
        # Compute forward
        layers = self.forward(x)
        
        # Compute gradients via backpropagation
        # Compute gradients for last layer
        yhat = layers[-1].a
        delta = layers[-1].f.backward_crossentropy(yhat, y) # simplified derivative
        dW[-1] = delta @ layers[-2].a.T
        db[-1] = delta
        
        for l in range(len(self.W)-2, -1, -1):
            # Get gradients and delta for layer l
            dWl, dbl, delta = layers[l].backward(delta, self.W[l+1], self.b[l+1])
            
            # Update layer gradients
            dW[l] = dWl
            db[l] = dbl
        
        return dW, db
    
    #=============================
    # SGD Update
    '''
    Input:
        x:  training input
        y:  label
        lr: learning rate
    Return: NA
    '''
    #=============================
    def sgd_update(self, x, y, lr):
        x = x.reshape((len(x)**2, 1)) # reshape 2d data to be (w^2,1)
        y = y.reshape((len(y), 1))
        #print(x.shape)
        #print(y.shape)
        
        dW, db = self.backprop(x, y)
        
        # Update params with grads
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]
    
    #=============================
    # Train network
    '''
    Input:
        x_train: training data
        y_train: training labels
        x_val:   validation data
        y_val:   validation labels
        epochs:  times to train over entire training data set
        lr:      learning rate
    Return: NA
    '''
    #=============================
    def train(self, x_train=None, y_train=None, x_val=None, y_val=None, epochs=1, lr=0.001):
        print('Training over {} epochs. Learning rate={}.'.format(epochs, lr))
        print()
        T = len(y_train)
        
        # Iterate over epochs
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch+1))
            
            # Iterate over entire training data set
            for t in range(T):
                self.sgd_update(x_train[t], y_train[t], lr)
                
                #if t % 10000 and t != 0:
                if t%10000 == 0:
                    self.evaluate(x_val, y_val)
        
        self.evaluate(x_val, y_val)
