
import numpy as np
import pandas as pd

file_weights1 	= "weight1.csv"
file_weights2 	= "weight2.csv"

wg1     = pd.read_csv(file_weights1, parse_dates=True)
wg2     = pd.read_csv(file_weights2, parse_dates=True)
wg1     = wg1.loc[0:nHidden,str(0):str(nInput)]
wg2     = wg2.loc[0:nOutput,str(0):str(nHidden)]

w1      = wg1.values
w2      = wg2.values
