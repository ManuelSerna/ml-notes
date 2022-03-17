import numpy as np


class LRGD:
    ''' Linear Regression Classifier class
    '''
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        ''' Constructor
        param eta: (number) learning rate/step
        param n_iter: (int) number of times to iterate over all data samples X
        param random_state: seed for RNG
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Parameters
        self.beta = None
        self.intercept = None
        
        # Loss
        self.losses = None # train error
    
    def fit(self, X, y):
        ''' Fit linear reg. classifier to data
        param X: (numpy ndarray) (N, p)-shaped array for N data samples for p predictors
        param y: (numpy ndarray) (N,)-shaped array for corresponding samples' outputs
        
        return: LRGD object ready for classification
        '''
        n = X.shape[0] # number of samples
        p = X.shape[1] # number of coefficients for each predictor
        rgen = np.random.RandomState(self.random_state)
        self.beta = rgen.normal(loc=0.0, scale=0.01, size=p)
        self.intercept = np.array([0.])
        self.losses = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            dE = y - output

            # Update parameters of Linear Regression
            self.beta += self.eta * 2.0 * X.T.dot(dE) / n
            self.intercept += self.eta * 2.0 * dE.mean()
            
            loss = (dE**2).mean()
            self.losses.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.beta) + self.intercept

    def predict(self, X):
        '''
        param X: (numpy ndarray) (N, p)-shaped array for N data samples for p predictors
        return: (ndarray)
        '''
        return self.net_input(X)
    
    def evaluate(self, X_val, y_val):
        ''' Evaluate model uisng validation (test) data
        
        param: X_val: validation data samples
        param: y_val: validation ground truths
        
        return: validation error/loss (correctly predicted/total)
        '''
        pred = self.net_input(X_val)
        diff = y_val - pred
        return (diff**2).mean()
    
    def print_errors(self, X_val, y_val):
        ''' Print training and validation error
        
        param: X_val: validation data samples
        param: y_val: validation ground truths
        
        return: NA
        '''
        print('(Avg) Training Error: {:.6f}'.format(np.mean(self.losses)))
        print('Validation Error:     {:.6f}'.format(self.evaluate(X_val, y_val)))
