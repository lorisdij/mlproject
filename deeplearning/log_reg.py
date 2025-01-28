import numpy as np

"""
    Objectif : créer un algorithme capable de prédire qui est malade(1)/non-malade(0) à partir de données x1 x2 en ajustant le poids des coefficient à l'aide d'une descente du gradient
"""

def init_variables():
    '''
        génerer des poids de coefs de base aléatoires ainsi qu'un biai nul    
    '''
    bias=0
    weights=np.random.randn(2)
    print('--weights--')
    print(weights)

    return bias, weights

def dataset():
    '''
        generate dataset : taux de protéine et glucides dans l'alimentation
    '''
    ligne_par_class=10
    healthy=np.random.randn(ligne_par_class,2)+[5,5]
    sick=np.random.randn(ligne_par_class,2)
    features=np.vstack([healthy,sick])

    targets=np.concatenate((np.zeros(ligne_par_class),np.zeros(ligne_par_class)+1))
    print('--features--')
    print(features)
    print('--targets--')
    print(targets)

    return features, targets

def pre_activation(features,weights,bias):
    '''
        effectue la pré-activation : calcul de la sortie en fonction des features et de leur weights associés + biai
    '''
    z=np.dot(features,weights) + bias
    print('--pre-activation--')
    print(z)
    return z

def activation(z):
    '''
        On applique la fonction logarithmique sur le résultat de la pré-activation
    '''
    y=1/(1+np.exp(-z))
    print('--y--')
    print(y)
    return y

def predict(y):
    '''
        On obtient en sortie ce qui est prédit donc 0 : non-malade ou 1 : malade (en arrondissant la resultat de la fonction d'activation)
    '''
    prediction=np.round(y)
    print('--prediction--')
    print(prediction)
    return prediction


if __name__=='__main__':
    features, targets = dataset()
    bias, weights = init_variables()
    z=pre_activation(features,weights,bias)
    y=activation(z)
    predict(y)
