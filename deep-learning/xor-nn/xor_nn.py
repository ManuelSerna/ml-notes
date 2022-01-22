#*********************************
'''
XOR Neural Network v2
- Objective: create feed forward neural network that learns the XOR function.
- Structure:
    input layer (l=0): 2 inputs
    hidden layer (l=1): 2 activations
    output layer (l=2): 1 activation/prediction

- Variables:
    Weights:       	Contained in a list called 'W' where each element is a (n x m) array and W[l, i, j] is the weight for a particular activation i in layer l which is connected to activation in layer l-1.
    Biases: 		Contained in a list called 'b' where each element is a (n x 1) array and b[l, i] is the bias for activation i in layer l.
    Weighted Sums: 	Contained in a list called 'Z' where each element is a (n x 1) array and Z[l, i] is the weighted sum for some activation i in layer l with respect all connected activations in layer l-1.
    Activations: 	Contained in a list called 'A' where each element is a (n x 1) array and A[l, i] is the weighted sum in Z[l, i] put through some activation function.
    Inputs: 		Contained in a (n x 1) list called 'X' where each element X[i] is some input.
    Labels: 		Contained in a (n x 1) list called 'Y' where each element Y[i] is the label corresponding to input X[i].
    
- Activation Function: sigmoid for l=1 and l=2.
'''
#*********************************

import numpy as np
import random

#---------------------------------
# XOR function
'''
Domain: -0.5 < x or y < 0.5, x or y=0
                                x or y=1 otherwise
Return: (x XOR y)
'''
#---------------------------------
# With the defined range, force x and y to be 0 or 1
def get_binary(z):
    if z > -0.5 and z < 0.5:
        return 0
    else:
        return 1

def xor(x, y):
    a = get_binary(x)
    b = get_binary(y)
    if a == b:
        return 0
    else:
        return 1

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
def intialize(s):
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
    - x: single input array (2 x 1)
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
    - x: single input (2 x 1)
    - y: single input label (1 x 1)
Return:
    - dW: calculated change for W
    - db: calculated change for b
'''
#---------------------------------
def back_prop(W, b, Z, A, s, x, y):
    L = len(s) - 1 # index of the last layer
    
    dW = []
    db = []
    dA = [] # use to keep track of activation partial derivatives (so calcs are not exponential)

    # Zero-init arrays partial derivative lists
    for l in range(L):
        dW.append(np.zeros(W[l].shape))
        db.append(np.zeros(b[l].shape))
        dA.append(np.zeros(A[l].shape))
    
    # let Y contain our only label for this particular network
    Y = [y]

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
# Predict label on test input
'''
Input:
    - x: single input (2 x 1) array
    - model: Python dictionary containing the optimized W and b
    - s: list where each element represents the number of activations in each layer l
Return:
    - scalar prediction value (0 or 1)
'''
#---------------------------------
def predict(x, model, s):
    W = model['W']
    b = model['b']
    
    Z, A = forward_prop(W, b, x, s)
    y = A[-1][0]
    p = get_binary(y)
    
    return p

#---------------------------------
# Train model
'''
Input:
    - X: list of training inputs
    - Y: list of training labels (correct outputs)
    - epochs: number of times the entire data set will be iterated over
    - s: list where each element represents the number of activations in each layer l
Return:
    - model: Python dictionary containing the optimized W and b
'''
#---------------------------------
def train(X, Y, epochs, s):
    W, b = intialize(s)
    n = len(Y)
    
    for epoch in range(epochs):
        for i in range(n):
            x = np.array([[X[0][i]], [X[1][i]]]) # insert one input at a time
            y = np.array([Y[i]])
            
            Z, A = forward_prop(W, b, x, s)
            dW, db = back_prop(W, b, Z, A, s, x, y)
            W, b = update(W, b, dW, db)
    
    model = {}
    model['W'] = W
    model['b'] = b
    
    return model

#---------------------------------
# Evaluate model for accuracy
'''
Input: 
    - model: Python dictionary containing the optimized W and b
    - s: list where each element represents the number of activations in each layer l
Return: NA
'''
#---------------------------------
def evaluate(model, s):
    # Setup
    W = model['W']
    b = model['b']
    
    tot_pts = 1000
    tot_correct = 0
    counter = 0
    
    X_test = [] # test data inputs
    Y_test = [] # test data labels
    
    # Generate test data
    for i in range(tot_pts):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        X_test.append((x, y))
        Y_test.append(xor(x, y))
    
    # Make predictions on test data
    while counter < tot_pts:
        x = X_test[counter][0]
        y = X_test[counter][1]
        x_test = np.array([[x], [y]])
        
        p = predict(x_test, model, s) # prediction
        yhat = Y_test[counter] # actual label
        
        # Incrase total correct if prediction matches true label
        if p == yhat:
            tot_correct += 1
            
        counter += 1
    
    # Print accuracy to console
    print('Accuracy: {}/{} = {}%'.format(tot_correct, tot_pts, tot_correct/tot_pts * 100))

# Driver code
s = [2, 2, 1] # number of neurons for each layer
X = [[0, 0, 1, 1], [0, 1, 0, 1]] # training data
Y = [0, 1, 1, 0] # training labels
epochs = 1000

# Since this is a small network, we can print out several accuracies of n models
n = 10
for i in range(n):
    model = train(X, Y, epochs, s)
    evaluate(model, s)