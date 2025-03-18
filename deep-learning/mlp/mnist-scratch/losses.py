#*********************************
# Losses module
# Author: Manuel Serna-Aguilera
#*********************************

import numpy as np

# Compute cross entropy loss
#=================================
'''
Input:
    y: true labels vector (one-hot)
    yhat: predictions (class probabilities)
'''
#=================================
def cat_cross_entropy(y, yhat):
    loss = -1.0 * np.sum(np.multiply(y, np.log(yhat)))
    return loss
