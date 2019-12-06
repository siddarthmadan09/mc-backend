import glob
import os
import sys
import pickle
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from trainModels import extractFeatures


def testSample(testSampleFile, username, classifier):
    identity = username.split('_')[1]
    featureSet = extractFeatures(testSampleFile)
    featureSet = featureSet.reshape(1,len(featureSet))
    path = os.getcwd()
    path1 = path + os.sep + 'Models' + os.sep
    dictModels= {'Naive-Bayes': 'modelNaiveBayes.dat', 'KNN': 'modelKNN.dat', 'SVM': 'modelSVM.dat', 'LogisticRegression': 'modelLR.dat'}

    modelFile = path1 + dictModels[classifier]
    with open(modelFile, 'rb') as pickleFile:
        model = pickle.load(pickleFile)

    prediction = model.predict(featureSet)
    if int(prediction[0]) == int(identity):
        return True
    else:
        return False

# if __name__ == "__main__":
#     x = testSample("/home/akshay/Documents/MC_Project/DataPerPerson/person_19.csv","/home/akshay/Documents/MC_Project/Models/modelKNN.dat")
#     print(x)
#     x = testSample("/home/akshay/Documents/MC_Project/DataPerPerson/person_19.csv","/home/akshay/Documents/MC_Project/Models/modelLR.dat")
#     print(x)
#     x = testSample("/home/akshay/Documents/MC_Project/DataPerPerson/person_19.csv","/home/akshay/Documents/MC_Project/Models/modelNaiveBayes.dat")
#     print(x)
#     x = testSample("/home/akshay/Documents/MC_Project/DataPerPerson/person_19.csv","/home/akshay/Documents/MC_Project/Models/modelSVM.dat")
#     print(x)
    