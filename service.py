#!/usr/bin/python
# This script contains the core computation logic
#
import numpy as np
import pandas as pd
from _operator import index


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


def meanAndSd(s):
    return s.describe()["mean"], s.describe()["std"]

def statistics(df, DATA_COLUMNS):    
    index = ['Mean', 'Median', 'StdDev']
    stats = pd.DataFrame(np.random.randn(3, 9), columns=DATA_COLUMNS, index=index);
    for column in DATA_COLUMNS:
        s = df[column]
        stats[column]['Mean'], stats[column]['StdDev'] = meanAndSd(s)
        stats[column]['Median'] = stats[column].median()
    return stats

def getMeanVector(df, DATA_COLUMNS):
    stat = statistics(df, DATA_COLUMNS)
    return stat['Mean':'Mean'].as_matrix(DATA_COLUMNS)

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
    meanNd1 = np.arange(ds.shape[1]);meanNd1.fill(0);meanNdCntr1 = 0
    meanNd2 = np.arange(ds.shape[1]);meanNd2.fill(0);meanNdCntr2 = 0
    indx_class_map = dict()
    classMap = {'M1':list(), 'M2':list(), 'MND1':meanNd1, 'MND2':meanNd2, 'ID_CL_MAP':indx_class_map}    
    for indx in range(ds.shape[0]):
        d1 = euclideanDistance(mean1, ds[indx])
        d2 = euclideanDistance(mean2, ds[indx])
        if d1 < d2:
            classMap['M1'].extend([indx])
            meanNd1 = ds[indx] + meanNd1
            meanNdCntr1 = meanNdCntr1 + 1
            indx_class_map[indx] = '2'
        elif d2 < d1:
            classMap['M2'].extend([indx])
            meanNd2 = ds[indx] + meanNd2
            meanNdCntr2 = meanNdCntr2 + 1
            indx_class_map[indx] = '4'
        else:
            print('WARN :', 'Euclidean distance is same from means %s,%s \n from vector %s, assigning arbitrarily'\
                   % (mean1, mean2, ds[indx]))
            if np.random.random_integers(0, 1) == 1:
                classMap['M1'].extend([indx])
                meanNd1 = ds[indx] + meanNd1
                meanNdCntr1 = meanNdCntr1 + 1
                indx_class_map[indx] = '2'
            else:
                classMap['M2'].extend([indx])
                meanNd2 = ds[indx] + meanNd2
                meanNdCntr2 = meanNdCntr2 + 1
                indx_class_map[indx] = '4'
    # print('****************************END OF Classify*****************************\n')
    classMap['MND1'] = meanNd1 / meanNdCntr1
    classMap['MND2'] = meanNd2 / meanNdCntr2
    classMap['ID_CL_MAP'] = indx_class_map
    return classMap

def getClusterDataFrame(df, indexes):
    """
    This function creates a new data frame with the rows specified by the indexes
    """
    cluster = df.copy(deep=True)
    for indx in df.index.values:
        if indx not in indexes:
            cluster.drop(indx, inplace=True)
    return cluster

def generateReport(df, classMap, limit=21):
    report = pd.DataFrame(np.random.randn(limit, 3), columns=['ID', 'Class', 'Predicted_Class']);
    for indx in range(0, limit):
        report['ID'][indx] = df.iloc[indx]['Scn']
        report['Class'][indx] = df.iloc[indx]['CLASS']
        report['Predicted_Class'][indx] = classMap['ID_CL_MAP'][indx]
    return report