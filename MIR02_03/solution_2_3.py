#!/usr/bin/env python

'''
Multimedia Information Retrieval (WS 2011)
Homework 2: nearest_neighbor implementation

Created on Nov 5, 2011

@author: federico raue
@contact: federico.raue@gmail.com
'''

import numpy
import pylab
from ex2 import read_dataset
from scipy.spatial.distance import euclidean



def show_scatter(data, labels, test_data, test_labels, error_rate):
    index_class_0 = numpy.where(labels == 0)[0]
    index_class_1 = numpy.where(labels == 1)[0]
    
    
    pylab.figure(1)
    pylab.subplot(121)
    pylab.scatter(data[index_class_1,0], data[index_class_1,1],c='b', marker='o')
    pylab.scatter(data[index_class_0,0], data[index_class_0,1],c='r', marker='o')
    pylab.title("training set: " + str(data.shape[0]) + " data points, " + str(len(index_class_0)) + \
                " class_0, " + str(len(index_class_1)) + " class_1" )
    pylab.xlabel("X")
    pylab.ylabel("Y")
    
    index_class_0 = numpy.where(test_labels == 0)[0]
    index_class_1 = numpy.where(test_labels == 1)[0]

    pylab.subplot(122)
    pylab.scatter(test_data[index_class_1,0], test_data[index_class_1,1],c='b', marker='o')
    pylab.scatter(test_data[index_class_0,0], test_data[index_class_0,1],c='r', marker='o')
    pylab.title("test set: " + str(test_data.shape[0]) + " data points, " + str(len(index_class_0)) + \
                " class_0, " + str(len(index_class_1)) + " class_1" )
    pylab.xlabel("X")
    pylab.ylabel("Y")
    
   
    
    pylab.figure(2)
    pylab.title("Error Rate")
    pylab.xlabel("K")
    pylab.ylabel("Error(%)")
    ind = numpy.array([1,10,20, 30, 40, 50])
    pylab.plot(ind, error_rate)
    pylab.grid(True)

    pylab.show()


def nearest_neighbor(data, labels, test_data, test_labels, number_k):
    c = numpy.zeros(len(test_labels), dtype=int)
    
    for i in range(len(test_data)):
        dist = numpy.zeros((len(data), 2), dtype=float)
        dist[:, 1] = labels[:]
        
        for j in range(len(dist)):
            dist[j,0] = euclidean (test_data[i], data[j])
                        
        k_distance = dist[numpy.argsort(dist[:,0])[:number_k],1]
        k_0 = len(numpy.where(k_distance == 0)[0])
        k_1 = len(numpy.where(k_distance == 1)[0])
                
        if (k_0 < k_1):
            c[i] = 1
        else:
            c[i] = 0
        
    error_rate = 0
    
    for z in range(len(c)):
        if c[z] != test_labels[z]:
            error_rate += 1
    
    print error_rate,len(c)
    return error_rate/float(len(c))       

#===============================================================================#
#                         MAIN PROGRAM                                          #
#===============================================================================#

train_data, train_labels = read_dataset("train-large.txt")
#train_data, train_labels = read_dataset("train-small.txt")
test_data, test_labels = read_dataset("test.txt")

error_rate = numpy.array([0.,0.,0.,0.,0.,0.])

error_rate[0] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 1) 
error_rate[1] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 10)
error_rate[2] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 20)
error_rate[3] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 30)
error_rate[4] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 40) 
error_rate[5] = nearest_neighbor(train_data, train_labels, test_data, test_labels, 50) 

print error_rate
show_scatter(train_data, train_labels, test_data, test_labels,error_rate)




