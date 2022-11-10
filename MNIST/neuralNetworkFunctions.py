import numpy as np

def initialisation(dimensions):

    parametres = {}

    for couche in range(1, len(dimensions)):
        parametres['W' + str(couche)] = np.random.randn(dimensions[couche], dimensions[couche-1])
        parametres['b' + str(couche)] = np.zeros((dimensions[couche], 1))

    return parametres

def forward_propagation(X, parametres):

    activations = {'A0': X}
    C = len(parametres)//2
    for couche in range(1, C +1):
        Z = parametres['W' + str(couche)].dot(activations['A' + str(couche-1)]) + parametres['b'+str(couche)]
        activations['A'+str(couche)] = 1/ (1+np.exp(-Z))

    return activations

def back_propagation(X, y, parametres, activations):

    m = y.shape[1]
    gradients = {}
    C = len(parametres)//2
    dZ = activations['A' + str(C)] - y

    for couche in reversed(range(1, C+1)):
        gradients['dW'+str(couche)] = 1/m * np.dot(dZ, activations['A'+str(couche-1)].T)
        gradients['db'+str(couche)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if couche > 1:
            dZ = np.dot(parametres['W'+str(couche)].T, dZ) * activations['A' + str(couche-1)] * (1 - activations['A' + str(couche-1)])

    return gradients

def update(gradients, parametres, learning_rate):

    C = len(parametres)//2

    for couche in range(1,C+1):
        parametres['W'+str(couche)]-= learning_rate*gradients['dW'+str(couche)]
        parametres['b'+str(couche)]-= learning_rate*gradients['db'+str(couche)]

    return parametres