import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self,amountOfParameters,amountOfNeuronsW2,amountOfOutcomes):
        #Initial weights and biases for each neuron
        self.W1=np.random.randn(amountOfNeuronsW2,amountOfParameters)
        self.W2=np.random.randn(amountOfOutcomes,amountOfNeuronsW2)
        self.B1=np.random.randn(amountOfParameters,1)
        self.B2=np.random.randn(amountOfParameters,1)
    def oneHot():
        raise NotImplementedError
    def RELU():
        raise NotImplementedError
    def RELUd():
        raise NotImplementedError
    def softMax():
        raise NotImplementedError
    def forwardPropagation(self,input):
        raise NotImplementedError
    def backPropagation(self,outputError,learningRate):
        raise NotImplementedError
    
        
#Data
panda = pd.read_csv("train.csv")
data = np.array(panda)
m, n = data.shape
dataTest = data[0:1000].T
Ytest = dataTest[0]
Xtest = dataTest[1:n]

dataTrain = data[1000:m].T
Ytest = dataTrain[0]
Xtest = dataTrain[1:n]

iterations = 1000
#InitializeNeuralNetwork
network = NeuralNetwork(784,10,10)
#RunTheAlgorithm

