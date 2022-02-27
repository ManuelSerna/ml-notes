#*********************************
''' LDA Demo using sklearn on several generated datasets
Page: 
https://scikit-learn.org/0.16/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py

Manuel Serna-Aguilera
'''

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


def make_simple_linear_data():
    ''' Make simple 2-class dataset
    '''
    x, y = make_classification(
        n_features=2, 
        n_redundant=0, 
        n_informative=2, 
        random_state=1, 
        n_clusters_per_class=1
    )

    x += 2*np.random.uniform(size=x.shape)
    return (x, y)
    

def train_datasets(classifier, train_data):
    ''' Fit LDA classifier to data
    '''
    x_train, y_train = train_data
    classifier = LDA()
    classifier.fit(x_train, y_train)
    #return classifier


def lda_classify_datasets_demo(datasets):
    ''' Show LDA results on several datasets
    '''
    figure = plt.figure(figsize=(8, 6))
    i = 1

    for ds in datasets:
        mesh_step = 0.2
        x, y = ds # get dataset points
        
        # Split current dataset into training and test sets
        x = StandardScaler().fit_transform(x) # set mean=0 and set var=1 for input values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)

        x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
        y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step))
        
        # On top, show dataset only
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        ax = plt.subplot(len(datasets), 2, i)
        ax.scatter(
            x_test[:, 0], 
            x_test[:, 1], 
            c=y_test, 
            cmap=cm_bright, 
            alpha=0.6
        )
        ax.set_xticks(())
        ax.set_yticks(())
                
        # Below that, show results of classifer on dataset
        classifier = LDA()
        classifier.fit(x_train, y_train)
        
        if hasattr(classifier, "decision_function"):
            Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
        # Put the result into a color plot
        i += 1
        ax = plt.subplot(len(datasets), 2, i)

        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(
            x_test[:, 0], 
            x_test[:, 1], 
            c=y_test, 
            cmap=cm_bright, 
            alpha=0.6
        )
        ax.set_xticks(())
        ax.set_yticks(())
        
        i += 1
    
    # Plot datasets
    plt.show()


if __name__ == '__main__':
    # Create dummy datasets where number of variables/features/predictors is p=2
    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        make_simple_linear_data()
    ]
    
    lda_classify_datasets_demo(datasets)

    print('Done.')
