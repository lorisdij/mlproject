import numpy as np

def init_variables():
    """
        Init model variables (weights,bias)
    """
    weights= np.random.normal(size=2)
    bias=0
    return weights,bias


def get_dataset():
    """
        Method used to generate the dataset
    """
    #number of row per class
    row_per_class=5
    sick = np.random.randn(row_per_class,2) + np.array([-2,-2])
    healthy = np.random.randn(row_per_class,2) + np.array([2,-2])

    features = np.vstack([sick,healthy])
    targets = np.concatenate((np.zeros(row_per_class),np.zeros(row_per_class)+1))

    #print(targets)
    return features, targets

def pre_activation(features,weights,bias):
    """
        Compute pre activation
    """
    return np.dot(features,weights) + bias

def activation(z):
    """
        Compute activation
    """
    return 1/(1+np.exp(-z))
    


if __name__=='__main__':
    weights,bias=init_variables()
    features,targets=get_dataset()
    z=pre_activation(features,weights,bias)
    a=activation(z)
    #print(features)
    #print(targets)
    #print(weights)
    #print(bias)
    print(z)
    print(a)