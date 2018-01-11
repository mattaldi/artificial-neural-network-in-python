import numpy as np
import pandas as pd
out0  = np.array([7.4,	2.8,	6.1,	1.9])
label = ["setosa","versicolor","virginica"]

i = 4
nInputNodes  = i
h = 8
nHiddenNodes = h
j = 3
nOutputNodes = j

file_weights1 	= "weight1_new.csv"
file_weights2 	= "weight2_new.csv"

wg1     = pd.read_csv(file_weights1, header=None)
wg2     = pd.read_csv(file_weights2, header=None)

w1      = wg1.values
w2      = wg2.values

    

out1 = np.zeros([h])
out2 = np.zeros([j])


#hidden layer
for h in range(0,nHiddenNodes):
    sums = 0
    for i in range(0,nInputNodes):
        a       =   w1[h][i]  *  out0[i]
        sums    = sums + a
    out1[h]   =   1.0  /  (1.0  +  np.exp(-sums))

#output layer
for j in range(0,nOutputNodes):
    sums = 0
    for h in range(0,nHiddenNodes):
        a       =   w2[j][h]  *  out1[h]
        sums    = sums + a
        
    out2[j]   =   1.0  /  (1.0  +  np.exp(-sums))

mval = max(out2)


i = out2.tolist().index(mval)

print label[i]
