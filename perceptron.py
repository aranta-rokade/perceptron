import random
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

def main():

    #training constants
    epoch = 10000
    init_weights = [random.uniform(-100.0,100.0), random.uniform(-100.0,100.0), random.uniform(-100.0,100.0)]
    train_fileNames=["set1.train","set2.train", "set3.train", "set4.train", "set5.train",
                     "set6.train","set7.train", "set8.train", "set9.train", "set10.train"]
    test_fileName = "set.test"
    
    i=1
    boundaries = []
    for fileName in train_fileNames:
        
        #read training data
        print("=====================================")
        print("Training File : {0}".format(fileName))
        train_data = read_file(fileName)
        
        #learn weights using the perceptron algorigthm
        result = perceptron(epoch, train_data, init_weights)
        boundary = result.get("weights")
        print("Epochs", result.get("epochs"))
        print("Weights", boundary)
        boundaries.append(boundary[:])
        
        #read test data
        print("Test Data")
        test_data = read_file(test_fileName)        
        
        #test data on the learnt weights
        error_count = test(test_data, boundary)
        error_rate = error_count/len(test_data)
        print("Error Count : {0}".format(error_count))
        print("Error Rate : {0}".format(error_rate))
        
        #plot training & test data with the boundary
        #plt.subplot(5, 5, i*2)
        plt.figure(i)
        plt.subplot(2,1,1)
        plot_data(train_data, boundary, fileName)
        i += 1
        plt.subplot(2,1,2)
        plot_data(test_data, boundary, test_fileName)
    #plot test data with all the boundaries
    plt.figure(i)
    plot_test_data(test_data, boundaries, test_fileName)
    plt.show()

#function to read the data files
def read_file(fileName):
    with open(fileName, "r") as file:
        array = []
        for line in file:
            x = line.strip().split(' ')
            array.append([float(i) for i in x])
    return array

#function to calculate the activation function
def activation(data, weights):
    theta = weights[-1]
    activation = theta
    for i in range(len(data)-1):
        activation += weights[i]*data[i]
    return 1.0 if activation >= 0.0 else 0.0   
    
#function to implement perceptron to lear the weights
def perceptron(epoch, data, weights):
    for j in range(epoch):
        error_count = 0
        for d in data:
            actual_class = d[-1]
            predicted_class = activation(d, weights)
            error = actual_class - predicted_class
            weights[-1] += error
            for i in range(len(d)-1):
                weights[i] += error*d[i]
            if(error !=0):
                error_count += 1
        if(error_count == 0):
            break
    return {"weights" : weights, "epochs": j}

#function to test data using weights
def test(data, weights):
    error_count = 0
    for d in data:
        actual_class = d[-1]
        predicted_class = activation(d, weights)
        error = actual_class - predicted_class
        if(error !=0):
            error_count += 1
    return error_count

#Following functions are used to plot the data and boundary 

#Equation of boundary
#Returns y co-ordinate for a boundary given an x co-ordinate 
def boundary_equation(weights, x):
    return -1*(weights[-1] + weights[0]*x) / weights[1]

#plots the training data and its boundary
def plot_data(data, weights, title):
    plt.title(title)
    max_x=data[0][0]
    min_x=data[0][0]
    for d in data:
        color = 'blue' if d[-1] == 0 else 'red' 
        plt.plot(d[0], d[1], '.', color=color)
        min_x = min(min_x,d[0])
        max_x = max(max_x,d[0])

    x = np.arange(min_x, max_x, 1)
    plt.plot(x, boundary_equation(weights, x))

#plots the testing data and all the boundaries
def plot_test_data(data, boundaries, title):
    plt.title(title)
    max_x=data[0][0]
    min_x=data[0][0]
    for d in data:
        color = 'blue' if d[-1] == 0 else 'red' 
        plt.plot(d[0], d[1], '.', color=color)
        min_x = min(min_x,d[0])
        max_x = max(max_x,d[0])

    x = np.arange(min_x, max_x, 1)
    for b in boundaries:
        plt.plot(x, boundary_equation(b, x))

main()