import pandas as panda
import numpy as np
import pprint
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Global values aka file information change these when file input changes
NUM_OUTPUT_NODES = 2

def localFieldCalculation(bias, w, x, nodes):
    v = []
    for n in range(0, nodes):
        v.append(np.dot(x, w[n]) + bias[n])
    return v

def localFieldOutputLayerCalculation(bias, w, x, nodes):
    v = []
    for n in range(0, nodes):
        v.append(np.dot(x, w[n]) + bias[n])
    return v

# This is phi(v) = 1 / 1 + exp(-v)
def activationFunction(v, nodes):
    y = []
    for n in range(0, nodes):
        y.append((1/(1+(math.exp(-v[n])))))
    return y

def derivativeActivationFunction(y):
    return y*(1-y)

def computingErrors(d, y):
    return d - y

def computingDelta(e, y):
    delta = []
    for n in range(0, len(e)):
        delta.append(e[n] * derivativeActivationFunction(y[n]))
    return delta

def deltaBias(delta, previousDeltaBeta, learningRate, beta):
    return beta * previousDeltaBeta + learningRate * delta

def deltaWeights(delta, y, previousDeltaWeight, learningRate, beta):
    return beta * previousDeltaWeight + learningRate * delta * y

def computeHiddenLayerDelta(y, delta, weights, depth):
    return np.dot(delta, weights) * derivativeActivationFunction(y[depth])

def backPropagation(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes):

    previousDeltaWeights1 = []
    previousDeltaWeights2 = []
    previousDeltaBias1 = []
    previousDeltaBias2 = []
    sumSquaredError = []
    sumSquaredErrorTotal = 0

    numOfSamples = crossData.shape[0]

    for sample in range(0, numOfSamples):

        # --------- This is the Forward Pass --------------------
        # These local fields (v) and outputs (y) are for the first hidden layer
        # print(crossData.loc[sample,:numOfInputs])
        x = crossData[sample][:numOfInputs]
        v = localFieldCalculation(hiddenBias, weightsInputToHidden, x, numOfHiddenNodes)
        y = activationFunction(v, numOfHiddenNodes)
        # print('\nFirst Output Hidden Layer\n')

        # These local fields and outputs are for the output layer
        v2 = localFieldCalculation(outputBias, weightsHiddenToOutput, y, NUM_OUTPUT_NODES)
        y2 = activationFunction(v2, NUM_OUTPUT_NODES)
        # print('\nOutputLayer Outputs\n')
        # print(y2)
        # -------- BackPropagation ---------
        # calculating errors at output layer
        e2 = computingErrors(crossData[sample][numOfInputs:], y2)

        # calculating the change in errors
        deltaOutputLayer = computingDelta(e2, y2)

        # looping from 0 to number of hidden layer nodes
        # and appending the change in weights of hidden layer
        deltaInputLayer = []

        for n in range(0, numOfHiddenNodes):
            deltaInputLayer.append(
                computeHiddenLayerDelta(y, deltaOutputLayer, [weightsHiddenToOutput[0][n], weightsHiddenToOutput[1][n]], n))

        w2Delta = []

        for i in range(0, NUM_OUTPUT_NODES):
            w2DeltaNeuron = []
            for j in range(0, numOfHiddenNodes):
                # previousDeltaWeight is only 0 for first epoch
                # but neglecting it due to limitation of assignment
                if sample == 0:
                    w2DeltaNeuron.append(
                        deltaWeights(deltaOutputLayer[i], y[j], 0, learningRate, beta))
                else:
                    w2DeltaNeuron.append(
                        deltaWeights(
                            deltaOutputLayer[i], y[j],
                            previousDeltaWeights2[i][j],
                            learningRate,
                            beta))
            w2Delta.append(w2DeltaNeuron)

        # changing the values of output layer weights
        previousDeltaWeights2 = w2Delta
        for i in range(0, NUM_OUTPUT_NODES):
            weightsHiddenToOutput[i] += w2Delta[i]

        # Compute delta weights for each node in the hidden layer
        w1Delta = []
        for i in range(0, numOfHiddenNodes):
            w1DeltaNeuron = []
            for j in range(0, numOfInputs):
                if sample == 0:
                    w1DeltaNeuron.append(
                        deltaWeights(deltaInputLayer[i], x[j], 0, learningRate, beta))
                else:
                    w1DeltaNeuron.append(
                        deltaWeights(
                            deltaInputLayer[i], x[j],
                            previousDeltaWeights1[i][j],
                            learningRate,
                            beta))
            w1Delta.append(w1DeltaNeuron)

        # changing the values of hidden layer weights
        previousDeltaWeights1 = w1Delta
        for i in range(0, numOfHiddenNodes):
            weightsInputToHidden[i] += w1Delta[i]

        b2Delta = []
        for i in range(0, NUM_OUTPUT_NODES):
            if sample == 0:
                b2Delta.append(deltaBias(deltaOutputLayer[i], 0, learningRate, beta))
            else:
                b2Delta.append(
                    deltaBias(deltaOutputLayer[i], previousDeltaBias2[i], learningRate, beta))

        # calcuation of change in bias
        # and updating the values for num of nodes in output layer
        previousDeltaBias2 = b2Delta
        for i in range(0, NUM_OUTPUT_NODES):
            outputBias[i][0] += b2Delta[i]

        b1Delta = []
        for i in range(0, numOfHiddenNodes):
            if sample == 0:
                b1Delta.append(deltaBias(deltaInputLayer[i], 0, learningRate, beta))
            else:
                b1Delta.append(deltaBias(deltaInputLayer[i], previousDeltaBias1[i], learningRate, beta))

        # calcuation of change in bias and updating the values for num of nodes in output layer
        previousDeltaBias1 = b1Delta
        for i in range(0, numOfHiddenNodes):
            hiddenBias[i, 0] += b1Delta[i]

        sumOfErrors = 0
        for e in e2:
            sumSquaredErrorTotal += e**2
            sumOfErrors += e**2
        sumSquaredError.append(sumOfErrors)

    tempSumSquaredError = 0

    for e in sumSquaredError:
        tempSumSquaredError += e / 2

    sumSquaredError = tempSumSquaredError
    meanError = sumSquaredError / numOfSamples

    return  weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, sumSquaredError

def classifier(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, numOfHiddenNodes):

    output = []
    for sample in range(0, len(crossData)):

        x = crossData[sample]
        v = localFieldCalculation(hiddenBias, weightsInputToHidden, x, numOfHiddenNodes)
        y = activationFunction(v, numOfHiddenNodes)

        v2 = localFieldCalculation(outputBias, weightsHiddenToOutput, y, NUM_OUTPUT_NODES)
        y2 = activationFunction(v2, NUM_OUTPUT_NODES)

        output.append(y2.index(max(y2)))

    return output

def convergence(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes, minimumError):

    changeInError = 100000
    previousError = 0
    epochs = []
    sumSquaredError = []
    sumSquaredErrorPerEpoch = []
    while changeInError >= minimumError:

        # Runs Backprop with the values above until it runs once then it changes them and runs it again
        weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, sumOfMeanError = backPropagation(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes)

        # print(changeInError)

        if changeInError == 100000:
            epochs.append(1)
        else:
            epochs.append(epochs[-1] + 1)

        # Changing the sumOfMeanError to sum of squared error thus *314
        changeInError = abs(previousError - sumOfMeanError)
        previousError = sumOfMeanError

        sumSquaredErrorPerEpoch.append(previousError)

    # print('Hidden layer bias\n')
    # print(hiddenBias)
    # print('\nOutput Layer Bias\n')
    # print(outputBias)
    # print('\nWeights Hidden Layer\n')
    # print(weightsInputToHidden)
    # print('\nWeights Output Layer \n')
    # print(weightsHiddenToOutput)
    # print('\nMean of Squared errors: ', sumOfMeanError)
    # print(epochs)

    return sumSquaredErrorPerEpoch, epochs


def convergenceRandomized(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes, minimumError):

    changeInError = 100000
    previousError = 0
    epochs = []
    sumSquaredError = []
    sumSquaredErrorPerEpoch = []
    while changeInError >= minimumError:

        # Runs Backprop with the values above until it runs once then it changes them and runs it again
        weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, sumOfMeanError = backPropagation(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes)

        np.random.shuffle(crossData)
        # print(changeInError)

        if changeInError == 100000:
            epochs.append(1)
        else:
            epochs.append(epochs[-1] + 1)

        # Changing the sumOfMeanError to sum of squared error thus *314
        changeInError = abs(previousError - sumOfMeanError)
        previousError = sumOfMeanError

        sumSquaredErrorPerEpoch.append(previousError)

    # print('Hidden layer bias\n')
    # print(hiddenBias)
    # print('\nOutput Layer Bias\n')
    # print(outputBias)
    # print('\nWeights Hidden Layer\n')
    # print(weightsInputToHidden)
    # print('\nWeights Output Layer \n')
    # print(weightsHiddenToOutput)
    # print('\nMean of Squared errors: ', sumOfMeanError)
    # print(epochs)

    return sumSquaredErrorPerEpoch, epochs


def convergenceMultipleLearningRates(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes):

    sumSquaredError = []
    epochs = []
    index = 0
    minimumError = .001

    plt.xlabel("Number of Epochs")
    plt.ylabel("Sum of Squared Error")

    for alpha in learningRate:

        tempcrossData = np.array(crossData)
        tempweightsInputToHidden = np.array(weightsInputToHidden)
        tempweightsHiddenToOutput = np.array(weightsHiddenToOutput)
        temphiddenBias = np.array(hiddenBias)
        tempoutputBias = np.array(outputBias)

        output = []
        print("\nStarting BackProp until convergence on: ", alpha)

        output = convergence(tempcrossData, tempweightsInputToHidden, tempweightsHiddenToOutput, temphiddenBias, tempoutputBias, alpha, beta, numOfInputs, numOfHiddenNodes, minimumError)
        sumSquaredError.append(output[0])
        epochs.append(output[1])

        plt.plot(epochs[index], sumSquaredError[index], label=f'α = {alpha}')
        index += 1

    # print(epochs)
    # for i in range(0, len(epochs)):
    # plt.plot(sumSquaredError[i], epochs[i])


    # plt.plot(sumSquaredError[0], epochs[0])
    # plt.plot(sumSquaredError[1], epochs[1])
    # plt.plot(sumSquaredError[2], epochs[2])

    # handles, labels = plt.get_legend_handles_labels()
    # plt.legend(handles, labels)
    plt.legend(loc='upper right')
    plt.show()

def convergenceMultipleMomentums(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes):

    sumSquaredError = []
    epochs = []
    index = 0
    minimumError = .001

    plt.xlabel("Number of Epochs")
    plt.ylabel("Sum of Squared Error")

    for β in beta:

        tempcrossData = np.array(crossData)
        tempweightsInputToHidden = np.array(weightsInputToHidden)
        tempweightsHiddenToOutput = np.array(weightsHiddenToOutput)
        temphiddenBias = np.array(hiddenBias)
        tempoutputBias = np.array(outputBias)

        output = []
        print("\nStarting BackProp until convergence on: ", β)

        output = convergence(tempcrossData, tempweightsInputToHidden, tempweightsHiddenToOutput, temphiddenBias, tempoutputBias, learningRate, β, numOfInputs, numOfHiddenNodes, minimumError)
        sumSquaredError.append(output[0])
        epochs.append(output[1])

        plt.plot(epochs[index], sumSquaredError[index], label=f'β = {β}')
        index += 1

    # print(epochs)
    # for i in range(0, len(epochs)):
    # plt.plot(sumSquaredError[i], epochs[i])


    # plt.plot(sumSquaredError[0], epochs[0])
    # plt.plot(sumSquaredError[1], epochs[1])
    # plt.plot(sumSquaredError[2], epochs[2])

    # handles, labels = plt.get_legend_handles_labels()
    # plt.legend(handles, labels)
    plt.legend(loc='upper right')
    plt.show()

def generateWeights(numOfInputs, numOfNodes):

    weightsInputToHidden = []
    for columns in range(0, numOfNodes):
        weightsInputToHidden.append(np.random.uniform(-0.1, 0.1, numOfInputs))

    weightsInputToHidden = np.array(weightsInputToHidden)

    return weightsInputToHidden

def generateBias(numOfNodes):
    bias = []
    for n in range(0, numOfNodes):
        bias.append([np.random.uniform(-0.1, 0.1)])
    return np.array(bias)

def part4(numOfHiddenNodes):
    gaussianFile = 'Two_Class_FourDGaussians500.csv'
    data = panda.read_csv(gaussianFile, sep=',', header=None)

    data = data.values

    numOfInputs = 4
    numOfFolds = 5
    learningRate = 0.7
    beta = 0.3
    minimumError = 0.01

    weightsInputToHidden = generateWeights(numOfInputs, numOfHiddenNodes)
    weightsHiddenToOutput = generateWeights(numOfHiddenNodes, NUM_OUTPUT_NODES )
    hiddenBias = generateBias(numOfHiddenNodes)
    outputBias = generateBias(NUM_OUTPUT_NODES)

    foldedClass1, foldedClass2 = generateFolds(data, numOfInputs, numOfFolds)

    confusionMatrix = np.zeros(shape=(numOfFolds, 2, 2))

    data = []
    #print(foldableTestingData[0])
    for fold in range(0, numOfFolds):

        data = []
        # data = np.concatenate([foldedClass1[fold], foldedClass2[fold]])
        # data = np.array(data)

        for i in range(0, len(foldedClass1[fold])):
            data.append(foldedClass1[fold][i])
            data.append(foldedClass2[fold][i])

        testingData = np.array(data)
        np.random.shuffle(testingData)

        data = []
        index = 4
        while index >= 0:
            if index == fold:
                index -= 1
                continue

            for i in range(0, len(foldedClass1[fold])):
                data.append(foldedClass1[index][i])
                data.append(foldedClass2[index][i])
            index -= 1

        trainingData = np.array(data)
        np.random.shuffle(trainingData)

        # print(trainingData.shape)
        # print(weightsInputToHidden)
        weightsInputToHidden = np.array(weightsInputToHidden)

        sumSquaredErrorPerEpoch, epochs = convergence(trainingData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes, minimumError)
        #print('Sum Squared Error last Epoch: ',sumSquaredErrorPerEpoch[-1])

        output = classifier(testingData[:, :4], weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, numOfHiddenNodes)
        correctOutput = 0

        for i in range(0, len(testingData)):
            correctOutput = np.argmax(testingData[i, 4:])
            confusionMatrix[fold][correctOutput][output[i]] += 1

    for i in range(0, numOfFolds):
        print("\nConfusion Matrix at fold: ", i)
        for j in range(0, 2):
            print(confusionMatrix[i][j])
        # run back prop against 80%
        # run forward prop against 20%
        # generate confusion matrix based on result of the 20%

def generateFolds(data, numOfInputs, numOfFolds):

    # Assuming sorted by desired input aka data[:, numOfInputs]
    # pp = pprint.PrettyPrinter(indent=4)
    length = len(data[:, numOfInputs])
    splittingIndex = -1

    for i in range(0, length):
        if data[i, numOfInputs] == 1:
            splittingIndex = i
            break

    class1Data = data[:splittingIndex, :]
    class2Data = data[splittingIndex:, :]

    foldedClass1 = np.vsplit(class1Data, numOfFolds)
    foldedClass2 = np.vsplit(class2Data, numOfFolds)

    return foldedClass1, foldedClass2

if __name__ == "__main__":

    # Getting b1 from csv
    bias1File = 'b1 (11 nodes).csv'
    hiddenLayerBias = panda.read_csv(bias1File, sep=',', header=None)
    # print("Hidden Layer Bias")
    # print(hiddenLayerBias)
    # print('\n')

    hiddenBias = hiddenLayerBias.values
    # print(hiddenBias)

    # Getting b2 from csv
    bias2File = 'b2 (2 output nodes).csv'
    outputLayerBias = panda.read_csv(bias2File, sep=',', header=None)
    # print("Output Layer Bias")
    # print(outputLayerBias)
    # print('\n')

    outputBias = outputLayerBias.values

    # Getting w1 from csv
    weight1File = 'w1 (3 inputs - 11 nodes).csv'
    weightsLayer1 = panda.read_csv(weight1File, sep=',', header=None)
    # print("Weights 1st Layer ")
    # print(weightsLayer1)
    # print('\n')

    weightsInputToHidden = weightsLayer1.values

    # Getting w2 from csv
    weight2File = 'w2 (from 11 to 2).csv'
    weightsLayer2 = panda.read_csv(weight2File, sep=',', header=None)
    # print("Weights 2nd Layer")
    # print(weightsLayer2)
    # print('\n')

    weightsHiddenToOutput = weightsLayer2.values

    # Getting cross data from csv
    crossDataFile = 'cross_data (3 inputs - 2 outputs).csv'
    crossData = panda.read_csv(crossDataFile, sep=',', header=None)

    crossData = crossData.values

    # saving these to retrain on
    freshCrossData = np.array(crossData)
    freshWeightsInputToHidden = np.array(weightsInputToHidden)
    freshWeightsHiddenToOutput = np.array(weightsHiddenToOutput)
    freshHiddenBias = np.array(hiddenBias)
    freshOutputBias = np.array(outputBias)

    numOfHiddenNodes = 11
    numOfInputs = 3
    minimumError = 0.001

    # print(hiddenBias)
    # print('break')

    # ------- This is part 1 ----------

    learningRate = 0.7
    beta = 0.3


    convergenceRandomized(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes, minimumError)

    newData = []
    for x in np.arange(-2.1, 2.1, .01):
        for y in np.arange(-2.1, 2.1, .01):
            z = np.random.uniform(-0.1, 0.1)
            newData.append([x,y,z])

    newData = np.array(newData)
    output = classifier(newData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, numOfHiddenNodes)

    fig = plt.figure()
    ax = fig.add_subplot(2.1, 2.1, 3)
    ax = fig.gca(projection='3d')

    colors = []
    for i in range(0, len(output)):
        if output[i] == 0:
            colors.append('blue')
        else:
            colors.append('red')

    # This graph could crash the window and thus halt the program would advise if
    # Running the other 3 parts to comment out the plot
    ax.scatter(xs=newData[:,0], ys=newData[:,1], zs=newData[:,2], c=colors)
    plt.show()

    # ------ End Part 1 ------

    # ------ Part 2 --------

    # Reseting the network to train again
    crossData = np.array(freshCrossData)
    weightsInputToHidden = np.array(freshWeightsInputToHidden)
    weightsHiddenToOutput = np.array(freshWeightsHiddenToOutput)
    hiddenBias = np.array(freshHiddenBias)
    outputBias = np.array(freshOutputBias)

    learningRate = [0.01, 0.2, 0.9, 0.7]
    beta = 0.3
    convergenceMultipleLearningRates(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes)

    # ------ End Part 2 ------

    # ------ Part 3 -------

    # Reseting the network to train again
    crossData = np.array(freshCrossData)
    weightsInputToHidden = np.array(freshWeightsInputToHidden)
    weightsHiddenToOutput = np.array(freshWeightsHiddenToOutput)
    hiddenBias = np.array(freshHiddenBias)
    outputBias = np.array(freshOutputBias)

    learningRate = 0.01
    beta = [0, 0.6]

    convergenceMultipleMomentums(crossData, weightsInputToHidden, weightsHiddenToOutput, hiddenBias, outputBias, learningRate, beta, numOfInputs, numOfHiddenNodes)

    # # ----- End Part 3 -----

    # ------ Part 4 --------
    numOfHiddenNodes = 8
    print(f"\n---- Running Cross Validation over Guassian Set with {numOfHiddenNodes} Hidden Nodes ---- \n")
    part4(numOfHiddenNodes)

    numOfHiddenNodes = 13
    print(f"\n---- Running Cross Validation over Guassian Set with {numOfHiddenNodes} Hidden Nodes ---- \n")
    part4(numOfHiddenNodes)

    numOfHiddenNodes = 15
    print(f"\n---- Running Cross Validation over Guassian Set with {numOfHiddenNodes} Hidden Nodes ---- \n")
    part4(numOfHiddenNodes)

    numOfHiddenNodes = 20
    print(f"\n---- Running Cross Validation over Guassian Set with {numOfHiddenNodes} Hidden Nodes ---- \n")
    part4(numOfHiddenNodes)

    # numOfHiddenNodes = 64
    # print(f"\n---- Running Cross Validation over Guassian Set with {numOfHiddenNodes} Hidden Nodes ---- \n")
    # part4(numOfHiddenNodes)