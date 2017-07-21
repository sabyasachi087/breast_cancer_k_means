#!/usr/bin/python
# This script contains the core computation logic
#
import numpy as np
import pandas as pd


def curate(df, fill="mean"):
    """ This function will fill the missing values with
        either mean , median or mode. Default is mean
    """
    if fill == "mean":        
        df.fillna(round(df.mean()), inplace=True)
    elif fill == "median":
        df.fillna(df.median(), inplace=True)
    elif fill == "mode":
        df.fillna(df.mode(), inplace=True)


def getRandomVectors(ds):
    """
    The function takes a numpy array as argument and 
    return two arbitrary rows
    """
    n = ds.shape[0]
    r1 = 0;r2 = 0
    while r1 == r2:
        r1 = np.random.random_integers(0, n - 1)
        r2 = np.random.random_integers(0, n - 1)
    return ds[r1], ds[r2]

def euclideanDistance(s1, s2):
    s = np.square(s1 - s2)
    return np.sqrt(np.sum(s))

def classify(ds, mean1, mean2):
    """
    This function classify each row in one of the category.
    The function returns dictionary of key means and value as
    list of indexes belong to this category.Use keys as mean1 -> M1 and mean2 -> M2
    """
    classMap = {'M1':list(), 'M2':list()}
    for indx in range(ds.shape[0]):
        d1 = euclideanDistance(mean1, ds[indx])
        d2 = euclideanDistance(mean2, ds[indx])
        if d1 < d2:
            classMap['M1'].extend([indx])
        elif d2 < d1:
            classMap['M2'].extend([indx])
        else:
            print('WARN :', 'Euclidean distance is same from means %s,%s \n from vector %s, assigning arbitrarily'\
                   % (mean1, mean2, ds[indx]))
            if np.random.random_integers(0, 1) == 1:
                classMap['M1'].extend([indx])
            else:
                classMap['M2'].extend([indx])
    print('****************************END OF Classify*****************************\n')
    return classMap

def getClusterDataFrame(df, indexes):
    """
    This function creates a new data frame with the rows specified by the indexes
    """
    cluster = df.copy(deep=True)
    for indx in indexes:
        cluster.drop(indx, inplace=True)
    return cluster
