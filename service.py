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
