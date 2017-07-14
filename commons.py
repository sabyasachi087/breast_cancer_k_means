#!/usr/bin/python
# This is an utility script for the k_means 
import numpy as np
import pandas as pd



BCANCER_WISCONSIN_DATASET = None

def init():
    df = pd.read_csv('Breast-cancer-wisconsin-1.csv') 
    df = df.fillna(0)
    # print(df.columns[[11,12]])
    df.drop(df.columns[[11, 12]], axis=1, inplace=True)
    global BCANCER_WISCONSIN_DATASET
    BCANCER_WISCONSIN_DATASET = df
    print('Commons Initialized')
    
    
init()
