#!/usr/bin/python
# Main script to launch the program
#
import commons as comm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import service as srvc
import dataplots as dp

df = comm.BCANCER_WISCONSIN_DATASET
srvc.curate(df)

def phase1():    
    dp.plotHisto(df, comm.DATA_COLUMNS)
    stats = comm.statistics(df)
    print(stats)

def phase2():
    print('Begin phase2')    
    ds = comm.getNumpyArray()
    s1, s2 = srvc.getRandomVectors(ds)   
    classMap = srvc.classify(ds, s1, s2)
    cluster1 = srvc.getClusterDataFrame(df, classMap['M1'])
    cluster2 = srvc.getClusterDataFrame(df, classMap['M2'])


if __name__ == '__main__':
    phase2()
