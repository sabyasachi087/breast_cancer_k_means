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
dp.plotHisto(df)




