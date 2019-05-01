# Importing libraries
import pandas as pd
import numpy as np
import math
import operator
from keras.datasets import mnist
import matplotlib.pyplot as plt
import random


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(y_train[10])

print(X_test.shape)
print(y_test.shape)
print(y_test[10])


def print_imgs(number_imgs):
    # plot 4 images as gray scale
    z=len(X_train)-1
    samples=random.sample(range(0, z), number_imgs)
    ind =1
    n_rows=int(number_imgs**(0.5))+1 
    n_cols=int(number_imgs**(0.5))+1

    for num in samples: 
        plt.subplot(n_rows,n_cols, ind)
        plt.imshow(X_train[num], cmap=plt.get_cmap('gray'))
        ind+=1
    # show the plot
    plt.show()
    print('Labels:',y_train[samples])

print_imgs(5)

# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2):
    distance = np.square(data1 - data2)
    return np.sqrt(np.sum(distance))


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')



# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# Defining our KNN model

def knn(trainSet,LabelTrainSet,testSet,LabelTestSet, k):
    cont=0    
    hit=0
    error=0
    length = testSet.shape[1]
    
    for testInstance,labelInstance in zip(testSet,LabelTestSet):
        distances = {}
        sort = {}
        
        # Calculating euclidean distance between each row of training data and test data
        for x in range(trainSet.shape[0]):
        
            dist = euclideanDistance(testInstance, trainSet[x])
            distances[x] = dist
       
        # Sorting them on the basis of distance
        sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
        neighbors = []
      
        # Extracting top k neighbors
        for x in range(k):
            neighbors.append(sorted_d[x][0])

        # Calculating the most freq class in the neighbors
        response = LabelTrainSet[neighbors]
        
        a=np.bincount((response))
        predict=np.argmax(a)
        
        if(predict==labelInstance):
            hit+=1;
        else:
            error+=1;
    return (hit/(hit+error))

#numbers of train's images 60000
samples=random.sample(range(0, 60000), 60000)

k=3
X_train2= X_train[samples]
y_train2 =y_train[samples]

#numbers of test's images: 1000
samples2=random.sample(range(0, 10000), 1000)

X_test2= X_test[samples2]
y_test2 =y_test[samples2]

acuracia=knn(X_train2, y_train2, X_test2,y_test2, k)
print(acuracia)

