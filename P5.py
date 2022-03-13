import numpy as np
from backpropagation.NeuralNetwork import NeuralNetwork
from backpropagation.Backpropagation import Backpropagation
from backpropagation.Sigmoid import Sigmoid

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) 
y = y/100

nn = NeuralNetwork(2,3,1)
nn.initalize_weights(True)

acivation_function = Sigmoid()
bp = Backpropagation(nn,5000,0.1,acivation_function)
bp.train(X,y)
bp.predict(X,y)
