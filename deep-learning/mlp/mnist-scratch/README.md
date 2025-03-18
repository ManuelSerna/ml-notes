# MNIST MLP
This is a very basic and very naive implementation of a multilayer perceptron (MLP). 
The code does things like instantiating layers very inefficiently, but there are 
many ways of implementing the tracking of parameters.

LaTeX notes and source (.tex) are included.


For the data, check out [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). 
Create a new folder called ```data``` and put the train and test data and labels files inside.

Pytorch, through the torchvision library, also has the dataset ready to download if the above source is removed.

```python
import torchvision
import torchvision.datasets as datasets

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=None)
```

There are also other copies available out on the web. In any case, this will require
re-implementing ```get_mnist()``` in ```data.py```
