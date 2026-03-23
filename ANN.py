import ast

import numpy as np      #auxaliary functions in case plotting is used, but generally not needed
import matplotlib as pl #in case of plotting cost function convergance
import random           #intended for assigning initial weights values randomly
import json

from matplotlib import pyplot as plt


def SingleNeuron(X, W): # single neuron, input consist of X and their weights W
    weightSum = W[0]                         #declaring the accumulating sum
    for i in range(len(X)):
        weightSum = weightSum + X[i] * W[i+1]#weighted sum for the activation function
    return Sigmoid(weightSum)                #a neuron's response to the X signals is the value of the excitation function

def Sigmoid(x):               #neuron's excitation function
    return 1/(1 + np.exp(-x)) #analytical expression

def SingleLayerNetwork(X, W):             #a single layer NN (there could be any numbers of neurons in it)
    row = len(W)                          #The weight's matrix has the dimensions those of the NN architecture( i. e., the number of neurons), var "row" is "row"
    col = len(W[0])                       #and the number of input signals into a single neuron, (not used)  the dimensions of the W matrix are row x st
    Y=[];                                 #array to store the responses of all neurons
    for i in range(row):                  #"running" through all neurons
        Y.append(SingleNeuron(X, W[i]))   #each time an individual neuron is excited, each excitation is then stored in array Y
    return Y                              #returning all responses of all neurons (stored in a single array Y)

def ErrorBackPropagation(XSet, W, NNteacherSet):#the training of the single layer network
    iterMax = 2000
    eps = 1e-6
    iterNr = 1
    step = 0.1
    TestSetElNum = len(XSet)
    auxProceed = True

    while auxProceed:
        max_derivative = 0

        for TestSetElInd in range(TestSetElNum):
            X = XSet[TestSetElInd]
            NNteacherAns = NNteacherSet[TestSetElInd]
            X1 = [1] + X
            NNPredAns = SingleLayerNetwork(X, W)

            for i in range(len(NNteacherAns)):
                derr = (NNteacherAns[i] - NNPredAns[i]) * NNPredAns[i] * (1 - NNPredAns[i])
                max_derivative = max(max_derivative, abs(derr))
                for j in range(len(W[0])):
                    W[i][j] = W[i][j] + derr * X1[j] * step

        auxProceed = (max_derivative > eps) and (iterNr < iterMax)
        iterNr += 1

    return W                                                     #returning the weights of the trained network

def ReadTrainingSets(path):
    XSet = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            if line.startswith('[') and line.endswith(']'):
                nums = line[1: -1].split(',')
                nums = [int(num) for num in nums]
                XSet.append(nums)
    return XSet

def BuildTeacher(NumberOfDigits, ExamplesPerDigit):
    Teacher = []
    for d in range(NumberOfDigits):
        vec = [0] * NumberOfDigits
        vec[d] = 1
        for _ in range(ExamplesPerDigit):
            Teacher.append(vec)
    return Teacher

def TrainDigits(XSet):
    #trains 10 neurons using NumberOfDigitsExamples
    #each Xset[i] is a pixel vector
    TotalNumberOfDigitExamples = len(XSet)
    NumberOfDigits = 10
    ## exeptionas if NumberOfDigitExamples % NumberOfDIGITS != 0 RUNTIMERROR

    ExamplesPerDigit = TotalNumberOfDigitExamples // NumberOfDigits

    InputSize = len(XSet[0])
    print(f"Each example has {InputSize} pixels/features.")
    print(f"Total examples: {TotalNumberOfDigitExamples}, {ExamplesPerDigit} per digit.")

    NNTeacherSet = BuildTeacher(NumberOfDigits, ExamplesPerDigit)

    W = []
    for _ in range(NumberOfDigits):
        NeuronWeights = [random.uniform(-1, 1)
                         for _ in range(InputSize + 1)]
        W.append(NeuronWeights)

    W = ErrorBackPropagation(XSet, W, NNTeacherSet)
    return W

def PredictDigit(X, W):
    outputs = SingleLayerNetwork(X, W)
    predicted = int(np.argmax(outputs))
    return outputs, predicted

def SaveWeights(W, path="TrainedWeights.txt"):
    with open(path, "w") as f:
        f.write("Output layer weights:\n")
        for neuron in W:
            plain = [float(w) for w in neuron]
            f.write(str(plain) + "\n")   # writes list like [0.12, -0.44, ...]
    print(f"Weights saved to {path}")

def LoadWeights(path="TrainedWeights.txt"):
    W = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Output layer"):
                continue
            if line.startswith("[") and line.endswith("]"):
                neuron_weights = ast.literal_eval(line)
                W.append(neuron_weights)
    print(f"Weights loaded from {path}")
    return W

## LEIDZIAM PROGRAMA
TrainingFile = "Training.txt"
XSet = ReadTrainingSets(TrainingFile)

W = TrainDigits(XSet)
SaveWeights(W, "TrainedWeights.txt")

# arba
#W = LoadWeights("TrainedWeights.txt")

Xtest = XSet[26]
outputs, digit = PredictDigit(Xtest, W)

print("Outputs:", outputs)
print("Predicted:", digit)



