#*********************************
# Losses module
# Author: Manuel Serna-Aguilera
#*********************************

import numpy as np

# Compute cross entropy loss
#=================================
'''
Input:
    yhat: predictions (one-hot)
    y: true labels vector (one-hot)
'''
#=================================
def cat_cross_entropy(yhat, y):
    loss = -1.0 * np.sum(np.multiply(y, np.log(yhat)))
    return loss
