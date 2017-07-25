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
    m4, m2, classMap = srvc.kmeans(comm.getNumpyArray(), 1500)
    print('\nM2 : %s' % (m2))
    print('M4 : %s\n' % (m4))  
    print(srvc.generateReport(df, classMap))

def phase3():
    print('Begin phase3 \n')   
    m4, m2, classMap = srvc.kmeans(comm.getNumpyArray(), 250)    
    print('\nM2 : %s' % (m2))
    print('M4 : %s\n' % (m4))  
    report = srvc.generateReport(df, classMap, limit=699)
    err_M, err_B = srvc.errorRate(report)
    print('Error rate Malign : %s, and Benign : %s' % (err_M , err_B))
    print('Ner Error rate : %s' % (err_M + err_B))
    
if __name__ == '__main__':
    phase3()
