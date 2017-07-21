#!/usr/bin/python
# This is an utility script for the k_means 
import numpy as np
import pandas as pd


DATA_COLUMNS = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
BCANCER_WISCONSIN_DATASET = None

def init():
    df = pd.read_csv('Breast-cancer-wisconsin-1.csv') 
    df = df.fillna(0)
    # print(df.columns[[11,12]])
    df.drop(df.columns[[11, 12]], axis=1, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    global BCANCER_WISCONSIN_DATASET
    BCANCER_WISCONSIN_DATASET = df
    print('Commons Initialized  \n')



def meanAndSd(s):
    return s.describe()["mean"], s.describe()["std"]

def statistics(df):    
    index = ['Mean', 'Median', 'StdDev']
    stats = pd.DataFrame(np.random.randn(3, 9), columns=DATA_COLUMNS, index=index);
    for column in DATA_COLUMNS:
        s = df[column]
        stats[column]['Mean'], stats[column]['StdDev'] = meanAndSd(s)
        stats[column]['Median'] = stats[column].median()
    return stats
 
def getNumpyArray():
     return BCANCER_WISCONSIN_DATASET.as_matrix(DATA_COLUMNS)
   
init()
