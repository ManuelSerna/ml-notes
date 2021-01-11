#*********************************
# Custom file to import MNIST data from subdirectory 'data'
# Author: Manuel Serna-Aguilera
#*********************************

# Imports
import idx2numpy # used for extracting mnist data
import matplotlib.pyplot as plt
import numpy as np


#=================================
# One-hot encode array for n classes
# Input: classes: number of classes (default=10); data: raw labels
#=================================
def one_hot_encode(classes=10, data=None):
    n = len(data)
    
    # Create zero-init one-hot array
    y = np.zeros((n, classes))
    
    # One-hot encode for each decimal label
    for i in range(n):
        y[i][data[i]] = 1
    
    return y

#=================================
# Get formatted MNIST images
#=================================
def get_mnist():
    # Define file names
    file_train_x = 'data/train-images-idx3-ubyte'
    file_train_y = 'data/train-labels-idx1-ubyte'
    file_test_x = 'data/t10k-images-idx3-ubyte'
    file_test_y = 'data/t10k-labels-idx1-ubyte'

    # Extract data into arrays
    # NOTE 1: All samples are already shuffled
    # NOTE 2: All image samples are numpy arrays of shape (28,28)
    x_train = idx2numpy.convert_from_file(file_train_x)
    y_train = idx2numpy.convert_from_file(file_train_y)
    x_test = idx2numpy.convert_from_file(file_test_x)
    y_test = idx2numpy.convert_from_file(file_test_y)
    
    # Normalize data into range [0,1]
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    # One-hot encode labels
    y_train = one_hot_encode(data=y_train)
    y_test = one_hot_encode(data=y_test)
    
    return x_train, y_train, x_test, y_test
