#*********************************
# Test custom LSTM
# Author: Manuel Serna-Aguilera
#*********************************

import data # use custom data import code
import mlp # use mlp (fully-connected network) 
import numpy as np

# Get MNIST data & one-hot encode
x_train, y_train, x_test, y_test = data.get_mnist()

# Train MLP model
s = [784, 16, 16, 10] # length of input at element 0, with subsequent elements denoting number of neurons for subsequent layers
n = 10 # classes
iterations = 5 # i.e. epochs
learning_rate = 0.001

model = mlp.MLP(s=s, n_classes=n)
model.train(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, epochs=5, lr=learning_rate)
