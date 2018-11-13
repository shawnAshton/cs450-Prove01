import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import random as rand
import math


# class Neuron:
#     def __init__(self):
#         self.threshHold = 0
#
#     def get_threshold(self):
#         return self.threshHold

# needs an attribute value and a list of weights...
class Node:
    def __init__(self, a_value, num_of_weights):
        self.value = a_value
        self.error = 0
        self.weights = []
        for i in range(0, num_of_weights):
            random_weight = rand.random()
            if rand.randint(1, 2) == 1:
                random_weight *= -1
            self.weights.append(random_weight)

    def get_weight(self, index):
        return self.weights[index]

    def set_weight(self, index, new_weight):
        self.weights[index] = new_weight

    def get_weight_length(self):
        return len(self.weights)

    def get_error(self):
        return self.error

    def set_error(self, error):
        self.error = error

    def get_value(self):
        return self.value

    def set_value(self, a_value):
        self.value = a_value


def get_iris_list():
    # The Iris dataset
    iris = datasets.load_iris()
    # Show the data (the attributes of each instance)
    scaler = StandardScaler()
    scaler.fit(iris.data)
    iris.data = scaler.transform(iris.data)
    return iris.data, iris.target


data, target = get_iris_list()
uniqueCountForTargets = len(np.unique(target))  # returns a array with the unique things..then find the amount in array.
biasAttributes = np.full(len(data), -1)
data = np.column_stack((data, biasAttributes))
layers = [3,2,3]


# create nodes for the inputs
dataNodeArray = []
inputNodes = []
for i in range(0, len(data[0])):
    inputNodes.append(Node(0, layers[0]))  # inputNodes.append("input NODE")
dataNodeArray.append(inputNodes)
# create all the nodes in all layers blank....
# first dimension is layer
# second dimension is node
for i in range(0, len(layers)):
    # print(i,' AND ', layers[i], '\n')
    nodes = []
    for j in range(0, layers[i]):
        # print('layer: ', layers[i], '  j: ', j)
        # nodes going to be created and put into an array...
        # this determines how many weights will be created for each node...
        if i+1 == len(layers):
            nodes.append(Node(0, 0))  # nodes.append("leafNode") aka output node
        else:
            nodes.append(Node(0, layers[i+1]))  # nodes.append("middle node")
            if j+1 == layers[i]:
                nodes.append(Node(-1, layers[i+1]))
    dataNodeArray.append(nodes)
print('inputs: ', len(dataNodeArray[0]))  # this is working and the neural net is built correctly..
print('1st: ', len(dataNodeArray[1])) # test out with alternative appends
print('2nd: ', len(dataNodeArray[2]))
print('3rd: ', len(dataNodeArray[3]))

learning_rate = 0.1
# LEARN!!!! LEARN!!!!!!
for learnLoop in range(0, 1000):
    # loop through the data(input values....) and get an answer
    for i in range(len(data)):  # loop over each row... which contains a set of input attributes... a different PROBLEM
        for j in range(len(data[i])):  # how many attributes do we have? loop that many times
            # add inputs as values to the input nodes(0 index of dataNodeArray)
            dataNodeArray[0][j].set_value(data[i][j])
        for layer in range(1, len(dataNodeArray)):
            # print(len(dataNodeArray[layer]))
            amount_of_nodes_in_layer = len(dataNodeArray[layer])-1
            if layer + 1 == len(dataNodeArray):
                amount_of_nodes_in_layer = len(dataNodeArray[layer])  # last layer
            for innerNode in range(0, amount_of_nodes_in_layer):
                # print('innerNode: ', innerNode, ' on layer: ', layer)
                # for every inner node... find the activation value...
                h = 0
                for nodes_in_past_layer in range(0,len(dataNodeArray[layer-1])):
                    # print('\t', nodes_in_past_layer)
                    h += dataNodeArray[layer-1][nodes_in_past_layer].get_weight(innerNode) * \
                         dataNodeArray[layer-1][nodes_in_past_layer].get_value()
                activation_value = 1 / (1 + math.e ** (-1 * h))
                dataNodeArray[layer][innerNode].set_value(activation_value)
        #         find error for all nodes...

        # error for output layer...
        last_layer = len(dataNodeArray) - 1
        correct_answer_index = target[i]
        for output in range(0, len(dataNodeArray[last_layer])):
            answer = 0
            if output == correct_answer_index:
                answer = 1
            aj = dataNodeArray[last_layer][output].get_value()
            dataNodeArray[last_layer][output].set_error(aj * (1 - aj) * (aj - answer))
        # find error for all inner nodes ... need to iterate backwards starting
        first_middle_layer = len(dataNodeArray) - 2
        for layer in range(first_middle_layer, 0, -1):
            # get the individual nodes from this layer...
            amount_of_nodes_in_layer = len(dataNodeArray[layer]) - 1
            amount_of_knodes_in_layer = len(dataNodeArray[layer + 1]) - 1
            for node in range(0, amount_of_nodes_in_layer):
                # got the activation value for this node...
                currentNodeError = 0
                sumOfWeightError = 0
                aj = dataNodeArray[layer][node].get_value()
                # get sum of weights jk and error of k.
                for knode in range(0, amount_of_knodes_in_layer):
                    sumOfWeightError += dataNodeArray[layer][node].get_weight(knode)\
                                        * dataNodeArray[layer + 1][knode].get_error()
                currentNodeError = aj * (1 - aj) * sumOfWeightError
                dataNodeArray[layer][node].set_error(currentNodeError)

        # NOW update the silly weights
        for layer in range(0,len(dataNodeArray) - 1): # -1 beacuse we don't do the last layer....
            amount_of_nodes_in_layer = len(dataNodeArray[layer]) - 1
            amount_of_knodes_in_layer = len(dataNodeArray[layer + 1]) - 1
            for node in range(0, amount_of_nodes_in_layer):
                # for every node... update the weights to the next layer...
                for nodeOnRight in range (0, amount_of_knodes_in_layer):
                    old_weight = dataNodeArray[layer][node].get_weight(nodeOnRight)
                    error_node_on_right = dataNodeArray[layer+1][nodeOnRight].get_error()
                    current_activation = dataNodeArray[layer][node].get_value()
                    new_weight = old_weight - learning_rate * error_node_on_right * current_activation
                    dataNodeArray[layer][node].set_weight(nodeOnRight, new_weight)

        # print results by finding the last layer
        # print('\nI IS:', i, '\n')
        best_Answer = -1
        best_Answer_index = -1
        for output in range(0, len(dataNodeArray[last_layer])):
            if dataNodeArray[last_layer][output].get_value() > best_Answer:
                best_Answer = dataNodeArray[last_layer][output].get_value()
                best_Answer_index = output
            # print(dataNodeArray[last_layer][output].get_value())
        # print('answer is at ', best_Answer_index, ' index')

# lets see if it LEARNED!!!!!!!
accuracy = 0
for i in range(len(data)):  # loop over each row... which contains a set of input attributes... a different PROBLEM
    for j in range(len(data[i])):  # how many attributes do we have? loop that many times
        # add inputs as values to the input nodes(0 index of dataNodeArray)
        dataNodeArray[0][j].set_value(data[i][j])
    for layer in range(1, len(dataNodeArray)):
        # print(len(dataNodeArray[layer]))
        amount_of_nodes_in_layer = len(dataNodeArray[layer]) - 1
        if layer + 1 == len(dataNodeArray):
            amount_of_nodes_in_layer = len(dataNodeArray[layer])  # last layer
        for innerNode in range(0, amount_of_nodes_in_layer):
            # print('innerNode: ', innerNode, ' on layer: ', layer)
            # for every inner node... find the activation value...
            h = 0
            for nodes_in_past_layer in range(0, len(dataNodeArray[layer - 1])):
                # print('\t', nodes_in_past_layer)
                h += dataNodeArray[layer - 1][nodes_in_past_layer].get_weight(innerNode) * \
                     dataNodeArray[layer - 1][nodes_in_past_layer].get_value()
            activation_value = 1 / (1 + math.e ** (-1 * h))
            dataNodeArray[layer][innerNode].set_value(activation_value)
    best_Answer = -1
    best_Answer_index = -1
    for output in range(0, len(dataNodeArray[last_layer])):
        if dataNodeArray[last_layer][output].get_value() > best_Answer:
            best_Answer = dataNodeArray[last_layer][output].get_value()
            best_Answer_index = output
        # print(dataNodeArray[last_layer][output].get_value())
    if(best_Answer_index == target[i]):
        accuracy += 1
    print('answer is at ', best_Answer_index, ' index with correct answer being: ', target[i], "  item number", i)
accuracy = accuracy / len(target)
print("Accuracy is: ", accuracy)

