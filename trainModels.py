import glob
import os
import sys
import pickle
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import json
import re
import pandas as pd


def extractFeatures(filename):
    userData = pd.read_csv(filename, header = None)
    sensors = 3
    featureSet = np.empty(shape=0)
    for column in np.arange(sensors):
        sensor_data = userData.iloc[:, column]
        sampleCount = len(sensor_data) 
        fftFeatures = fft(sensor_data)
        normFft = abs(fftFeatures/sampleCount)*2
        finalFft = normFft[0:int(sampleCount/2)]
        featureSet = np.concatenate((featureSet, finalFft[0:20]))
    return featureSet

def readFiles():
    sensorFeatures = np.zeros((1,60))
    labels = np.array([])

    path = os.getcwd()
    for file in glob.glob('DataPerPerson/*.csv'):
        m = re.split('[/ .]',file)
        label = int(m[1].split('_')[1])
        #print(label)
        featureSet = extractFeatures(file)
        featureSet = featureSet.reshape(1,len(featureSet))
        sensorFeatures = np.append(sensorFeatures, featureSet, axis=0)
        labels = np.append(labels, label)
    labels = labels.reshape(len(labels),1)
    #print(brainActList)
    sensorFeatures = sensorFeatures[1:]
    #print(brainActList)
    finalfeatures = np.append(sensorFeatures, labels, axis = 1)
    #print(finalfeatures)
    return finalfeatures

def modelNaiveBayes(train_data, test_data, train_labels, test_labels):
    nb = MultinomialNB()
    nb.fit(train_data, train_labels)
    path = os.getcwd() + os.sep + 'Models' + os.sep + 'modelNaiveBayes.dat'
    with open(path, 'wb') as p:
        pickle.dump(nb, p)
    score = nb.score(test_data, test_labels) * 100
    return score

def modelLR(train_data, test_data, train_labels, test_labels):
    lr = LogisticRegression(multi_class = 'auto', solver='liblinear')
    lr.fit(train_data, train_labels)
    path = os.getcwd() + os.sep + 'Models' + os.sep + 'modelLR.dat'
    with open(path, 'wb') as p:
        pickle.dump(lr, p)
    score = lr.score(test_data, test_labels) * 100
    return score

def modelSVM(train_data, test_data, train_labels, test_labels):
    sv = svm.SVC(kernel='linear', decision_function_shape='ovo', probability=True)
    sv.fit(train_data, train_labels)
    path = os.getcwd() + os.sep + 'Models' + os.sep +'modelSVM.dat'
    with open(path, 'wb') as p:
        pickle.dump(sv, p)
    score = sv.score(test_data, test_labels) * 100
    return score

def modelKNN(train_data, test_data, train_labels, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_labels)
    path = os.getcwd() + os.sep + 'Models' + os.sep + 'modelKNN.dat'
    with open(path, 'wb') as p:
        pickle.dump(knn, p)
    score = knn.score(test_data, test_labels) * 100
    return score

    
def generateModels():
    training_data = readFiles()
    #print(training_data.shape)
    data, labels = training_data[:,:-1], training_data[:, -1].astype(int)
    train_data, train_labels = data, labels
    test_data, test_labels = data[5:20], labels[5:20]
    # print(test_data[])
    # print(test_label)
    # print(labels)

    #train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.1, random_state = 40)
    modelResults = {}
    path = os.getcwd()

    modelResults["NaiveBayes"] = modelNaiveBayes(train_data, test_data, train_labels, test_labels)
    modelResults["LR"] = modelLR(train_data, test_data, train_labels, test_labels)
    modelResults["SVM"] = modelSVM(train_data, test_data, train_labels, test_labels)
    modelResults["KNN"] = modelKNN(train_data, test_data, train_labels, test_labels)

    print(modelResults)

    storeModelResults = path + os.sep + 'modelResults.json'
    with open(storeModelResults, 'w') as p:
        json.dump(modelResults, p)



if  __name__ == "__main__":
    generateModels()