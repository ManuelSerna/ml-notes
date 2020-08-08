#*********************************
'''
MNIST Neural Network v1.1

- Objective: Create SIMPLE feed forward neural network that can identify digits 0...9 that is optimized via gradient descent.

- Structure:
    - input layer (l=0):    784 inputs for every pixel in 28 x 28 input image
    - hidden layer 1 (l=1): 16 activations, sigmoid
    - hidden layer 2 (l=2): 16 activations, sigmoid
    - output layer (l=3):   10 activations, sigmoid

- Notation:
    - Weights:        Contained in a list called 'W' where each element is a (n x m) array and W[l, i, j] is the weight for a particular activation i in layer l which is connected to activation j in layer l-1.
    - Biases:         Contained in a list called 'b' where each element is a (n x 1) array and b[l, i] is the bias for activation i in layer l.
    - Weighted Sums:  Contained in a list called 'Z' where each element is a (n x 1) array and Z[l, i] is the weighted sum for some activation i in layer l with respect all connected activations in layer l-1.
    - Activations:    Contained in a list called 'A' where each element is a (n x 1) array and A[l, i] is the weighted sum in Z[l, i] put through some activation function.
    - Inputs:         Contained in a (n x 784) list called 'x' where each element x[i] is some input image. Training and test data will be put in lists called 'x_train' and 'x_test' respectively.
    - Labels:         Contained in a (n x 1) list called 'y' where each element y[i] is the label corresponding to input x[i]. Training and test data will be put in lists called 'y_train' and 'y_test' respectively.

- Activation Function: sigmoid for all layers.
'''
#*********************************

import json
import numpy as np
import random
import sys
import tensorflow as tf

#---------------------------------
# Get MNIST data--properly formated for this program
# Input: NA
# Return: training and test data
#---------------------------------
def get_data():
    # Pull data using Tensorflow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape x's to be 2D since the raw input contains a list of 2D arrays
    x_train = x_train.reshape(x_train.shape[0], 784) # shape = (60000, 784)
    x_test = x_test.reshape(x_test.shape[0], 784) # shape = (10000, 784)
    
    # Reshaping changes the nature of the data, turn it back to float 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalize after conversion to float
    x_train /= 255
    x_test /= 255
    
    return x_train, y_train, x_test, y_test

#---------------------------------
# Sigmoid function
# Input: weigted sum z
# Return: function result
#---------------------------------
def sigmoid(z):
    return 1/(1+np.exp(-z))

#---------------------------------
# Initialize parameters W and b
'''
Input:
    - s: list where each element l contains the number of neurons for each layer
Return:
    - Randomly generated lists for W and b
'''
#---------------------------------
def initialize(s):
    W = []
    b = []
    
    for l in range(len(s)-1):
        W.append(np.random.normal(0.0, 1.0, (s[l+1], s[l])))
        b.append(np.random.normal(0.0, 1.0, (s[l+1], 1)))

    return W, b

#---------------------------------
# Forward propagate
'''
Input:
    - W: weights list
    - b: biases list
    - x: single input array (784 x 1)
    - s: list where each element represents the number of activations in each layer l
Return:
    - Z: list of weighted sums
    - A: list of activations
'''
#---------------------------------
def forward_prop(W, b, x, s):
    Z = []
    A = []
    
    n = len(s)
    
    for l in range(n-1):
        if l == 0:
            z = W[0].dot(x)+b[0]
            a = sigmoid(z)
        else:
            z = W[l].dot(A[l-1])+b[l]
            a = sigmoid(z)
        Z.append(z)
        A.append(a)
    
    return Z, A

#---------------------------------
# Back propagate
'''
Input:
    - W: weights list
    - b: biases list
    - Z: list of weighted sums
    - A: list of activations
    - s: list where each element represents the number of activations in each layer l
    - x: single input (784 x 1)
    - y: single input label (1 x 1)
Return:
    - dW: calculated change for W
    - db: calculated change for b
'''
#---------------------------------
def back_prop(W, b, Z, A, s, x, y):
    L = len(s) - 1 # theoretically the index of the last layer
    
    dW = []
    db = []
    dA = []

    # Zero-init arrays partial derivative lists
    for l in range(L):
        dW.append(np.zeros(W[l].shape))
        db.append(np.zeros(b[l].shape))
        dA.append(np.zeros(A[l].shape))
    
    # Y will contain the desired labels of every output neuron given a specific training
    Y = []
    for i in range(s[L]):
        if i == y:
            Y.append(1.0)
        else:
            Y.append(0.0)

    # Back-propagate from output layer
    for l in range(L-1, -1, -1):
        for i in range(len(dW[l])):
            dz = sigmoid(Z[l][i]) * (1 - sigmoid(Z[l][i]))
            if l == L-1:
                dA[l][i] = 2 * (A[l][i] - Y[i])
            else:
                for k in range(len(dW[l+1])):
                    dA[l][i] += W[l+1][k][i] * sigmoid(Z[l+1][k]) * (1 - sigmoid(Z[l+1][k])) * dA[l+1][k]
            db[l][i] = dz * dA[l][i]
            for j in range(len(dW[l][i])):
                a = 0
                if l == 0:
                    a = x[j]
                else:
                    a = A[l-1][j]
                dW[l][i][j] = a * dz * dA[l][i]
    return dW, db

#---------------------------------
# Update weights and biases
'''
Input:
    - W: weights list
    - b: biases list
    - dW: calculated change for W
    - db: calculated change for b
Return:
    - Newly-updated W and b
'''
#---------------------------------
def update(W, b, dW, db):
    layers = len(W)
    
    for l in range(layers):
        W[l] = W[l] - dW[l]
        b[l] = b[l] - db[l]
    
    return W, b

#---------------------------------
# Train model
'''
Input:
    - x_train: training data (60,000 x 784)
    - y_train: training labels (60,000 x 1)
    - epochs: number of times the entire data set will be iterated over
    - s: list where each element represents the number of activations in each layer l
Return:
    - model: Python dictionary containing the optimized W and b
'''
#---------------------------------
def train(x_train, y_train, epochs, s):
    W, b = initialize(s)
    n = len(y_train)
    
    for epoch in range(epochs):
        for i in range(n):
            print('traninig sample {}'.format(i))

            # Insert one training sample and label at a time
            x = np.zeros((784, 1)) # need to make a 'column array'            
            for j in range(784):
                x[j, 0] = x_train[i][j]
            
            y = y_train[i]
            
            Z, A = forward_prop(W, b, x, s)
            dW, db = back_prop(W, b, Z, A, s, x, y)
            W, b = update(W, b, dW, db)
    
    model = {}
    model['W'] = W
    model['b'] = b
    
    return model

#---------------------------------
# Predict label on test input
'''
Input:
    - x: single input (2 x 1) array
    - model: Python dictionary containing the optimized W and b
    - s: list where each element represents the number of activations in each layer l
Return:
    - label of prediction with highest probability
'''
#---------------------------------
def predict(x, model, s):
    W = model['W']
    b = model['b']
    
    Z, A = forward_prop(W, b, x, s)
    
    max_prob = -1.0
    max_index = -1
    
    for k in range(len(A[-1])):
        if max_prob < A[-1][k]:
            max_prob = A[-1][k]
            max_index = k
    
    return max_index

#---------------------------------
# Evaluate model for accuracy
'''
Input: 
    - model: Python dictionary containing the optimized W and b
    - s: list where each element represents the number of activations in each layer l
    - x_test: test data (10,000 x 784)
    - y_test: test labels (10,000 x 1)
Return: NA
'''
#---------------------------------
def evaluate(model, s, x_test, y_test):
    n = len(y_test)
    tot_correct = 0
    
    # Make predictions on test data
    for i in range(n):
        x = np.zeros((784, 1)) # need to make a 'column array'
        
        for j in range(784):
            x[j, 0] = x_test[i][j]
        
        y = predict(x, model, s)
        #print(y_test[i], y)# DEBUG

        if y_test[i] == y:
            tot_correct += 1
    
    # Print accuracy to console
    print('Accuracy: {}/{} = {}%'.format(tot_correct, n, tot_correct/n * 100))

# Driver code
x_train, y_train, x_test, y_test = get_data()
epochs = 1
s = [784, 16, 16, 10] # highest acc = 82.39%

print('training...')
model = train(x_train, y_train, epochs, s)

print('testing...')
evaluate(model, s, x_test, y_test)