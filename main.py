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
    stats = srvc.statistics(df)
    print(stats)

def phase2():
    print('Begin phase2 \n')    
    ds = comm.getNumpyArray()
    s1, s2 = srvc.getRandomVectors(ds)   
    classMap = dict()
    for i in range(0, 1500):
        classMap = srvc.classify(ds, s1, s2)
        s1 = classMap['MND1']
        s2 = classMap['MND2']
    print()
    print('M2 : %s'%(s1))
    print('M4 : %s'%(s2))
    print()
    print(srvc.generateReport(df, classMap))

if __name__ == '__main__':
    phase2()
