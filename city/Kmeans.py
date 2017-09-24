from sklearn.cluster import KMeans

from sklearn.externals import joblib

import numpy as np

import time
from city import City_analize
import pandas as pd

import matplotlib.pyplot as plt


#!/usr/bin/env python
#coding:utf-8
from numpy import *
import time
import matplotlib.pyplot as plt

data = City_analize.City_analize().value_position_fushan()
data_full = City_analize.City_analize().value_fushan()
# print(data)

# data = pd.DataFrame(DATA,columns=['x','y'])
# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

                ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")

        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print
        "Sorry! Your k is too large! please contact Zouxy"
        return 1

        # draw all samples

    sort = []
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
        sort.append([i,markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()
    return sort

# print(len(data))
centroids, clusterAssment = kmeans(data, 3)
# print(centroids,clusterAssment)
t = showCluster(data, 3, centroids, clusterAssment)
t = np.array(t)
print(t,clusterAssment)
clusterAssment = np.array(clusterAssment)
classify_position = clusterAssment

### 区内划分
# 计算区内任务完成度以及区内的价格成本


#1.筛选出区域

def get_area_1_full():
    area_1 = []
    sum_ = 0
    for row in range(len(classify_position)):
        if float(clusterAssment[row][0]) == 0.00000000e+00:
            area_1.append([data_full[row][1],data_full[row][2],data_full[row][3],data_full[row][4],clusterAssment[row][1]])
            if float(data_full[row][4]) == 1.0:
                sum_ = sum_ + float(data_full[row][3])
    area_1 = np.array(area_1)
    # return len(area_1)
    return sum_ / len(area_1)

def get_area_2_full():
    area_2 = []
    sum_ = 0
    for row in range(len(classify_position)):
        if float(clusterAssment[row][0]) == 1.00000000e+00:
            area_2.append([data_full[row][1],data_full[row][2],data_full[row][3],data_full[row][4],clusterAssment[row][1]])
            if float(data_full[row][4]) == 1.0:
                sum_ = sum_ + float(data_full[row][3])
    area_2 = np.array(area_2)
    # return len(area_2)
    return sum_ / len(area_2)
def get_area_3_full():
    area_3 = []
    sum_ = 0
    for row in range(len(classify_position)):
        if float(clusterAssment[row][0]) == 2.00000000e+00:
            area_3.append([data_full[row][1],data_full[row][2],data_full[row][3],data_full[row][4],clusterAssment[row][1]])
            if float(data_full[row][4]) == 1.0:
                sum_ = sum_ + float(data_full[row][3])
    area_3 = np.array(area_3)
    # return {'full':area_3, 'sum':sum_, 'ave':(sum/area_3.__len__())}
    # return len(area_3)
    return sum_ / len(area_3)


# print(len(clusterAssment))
print(get_area_1_full())
print(get_area_2_full())
print(get_area_3_full())