import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def oneHot(Y):                                      #Creates oneHot array
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T                     
    return one_hot_Y
def ReLU(Z):
    return np.maximum(Z,0)                          #returns 0 or Z, whichever is bigger. Since Z is array, it iterates
def ReLUd(Z):
    return Z > 0                                    #return 1 or 0, depending if Z is bigger or not.
def softMax(Z):
    return np.exp(Z) / sum(np.exp(Z))               #This is just how softmax works. 
def get_predictions(A2):
    return np.argmax(A2, 0)
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


class NeuralNetwork:
    def __init__(self,amountOfParameters,amountOfNeuronsW2,amountOfOutcomes,learningRate):
        #Initial weights and biases for each neuron
        self.W1=np.random.rand(10,784)-0.5
        self.W2=np.random.rand(10,10)-0.5
        self.B1=np.random.rand(10,1)-0.5
        self.B2=np.random.rand(10,1)-0.5
        #InitialSetup
        self.learningRate = learningRate
        self.momentum=[1,1,1,1]
        self.averagesOfMomentumHistories=[1,1,1,1]     #0=W1, 1=W2, 2=B1, 3=B2
        self.amountOfOutcomes=amountOfOutcomes
    def forwardPropagation(self,X,Y):
        self.Z1=(self.W1 @ X) + self.B1           #Array X Array, + biases to matching neurons
        self.A1=ReLU(self.Z1)                   #
        self.Z2 = (self.W2 @ self.A1) + self.B2   #As above
        self.A2 = softMax(self.Z2)              #Turns values of Z to numbers between 0 and 1
    def backPropagation(self,X,Y):
        oneHotY = oneHot(Y)                     #Create a oneHot array
        #calculate how much off our biases and weights were, according to people smarter than me
        self.dZ2 = self.A2 - oneHotY
        self.dW2 = 1 / m * self.dZ2 @ self.A1.T
        self.db2 = 1 / m * np.sum(self.dZ2)
        self.dZ1 = self.W2.T.dot(self.dZ2) * ReLUd(self.Z1)
        self.dW1 = 1 / m * self.dZ1 @ X.T
        self.db1 = 1 / m * np.sum(self.dZ1)
    def updateParameters(self):
        self.W1-=(self.learningRate*self.dW1*self.momentum[0])
        print("dW1: "+(str)(np.amax(self.dW1)))
        self.W2-=(self.learningRate*self.dW2*self.momentum[1])
        self.B1-=(self.learningRate*self.db1*self.momentum[2])
        self.B2-=(self.learningRate*self.db2*self.momentum[3])
    def gradientDescent(self,iterations,X,Y):
        for i in range(iterations):
            self.forwardPropagation(X,Y)
            self.backPropagation(X, Y)
            self.updateParameters()
    def testing(self,X,Y,index):
        self.forwardPropagation(X,Y)
        prediction = get_predictions(self.A2)[index]
        label = Y[index]
        print("Actual: "+(str)(label))
        print("Predicted: "+(str)(prediction))
        image = X[:, index]
        image = image.reshape((28, 28)) * 255
        plt.imshow(image)
        plt.show()
    
        
#Data
panda = pd.read_csv("train.csv")
data = np.array(panda)
m, n = data.shape
#m=(int)(m/2)

dataTest = data[0:1000].T
Ytest = dataTest[0]
Xtest = dataTest[1:n]
Xtest= Xtest / 255.

dataTrain = data[1000:m].T
Ytrain = dataTrain[0]
Xtrain = dataTrain[1:n]
Xtrain = Xtrain/255.

#InitializeNeuralNetwork
iterations=200
network = NeuralNetwork(784,10,10,0.10)
#RunTheAlgorithm
network.gradientDescent(iterations,Xtrain,Ytrain)
#Testing
print("TESTING IN 5")
print("TESTING IN 4")
print("TESTING IN 3")
print("TESTING IN 2")
print("TESTING IN 1")
print("TESTING IN PROGRESS")
network.testing(Xtest,Ytest,1)