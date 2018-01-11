import numpy as np
import pandas as pd

fileCsv = 'iriscsv.csv'

datas   = pd.read_csv(fileCsv, header=None, parse_dates=True)
data    = datas.values

data1   = data[0:40,:]
data2   = data[50:90,:]
data3   = data[100:140,:]
dtrain  = np.concatenate((data1, data2, data3),axis=0)
fdtrain = dtrain[:,0:4]
ldtrain = dtrain[:,4]
out0    = fdtrain

data4   = data[40:50,:]
data5   = data[90:100,:]
data6   = data[140:150,:]
dtest   = np.concatenate((data4, data5, data6),axis=0)
fdtest  = dtest[:,0:4]
ldtest  = dtest[:,4]



##lnumtrain = []
##for i in range(0,len(ldtrain)):
##    labelnow = ldtrain[i]
##    if (labelnow == "Iris-setosa"):
##        lnumtrain = np.append(lnumtrain, 1)
##    if (labelnow == "Iris-versicolor"):
##        lnumtrain = np.append(lnumtrain, 2)
##    if (labelnow == "Iris-virginica"):
##        lnumtrain = np.append(lnumtrain, 3)
##
##    

x = 0
target = np.zeros([len(ldtrain),3])
for i in range(0,len(ldtrain)):
    labelnow = ldtrain[i]
    if (labelnow == "Iris-setosa"):
        target[x,0] = 1
    if (labelnow == "Iris-versicolor"):
        target[x,1] = 1
    if (labelnow == "Iris-virginica"):
        target[x,2] = 1
    x = x + 1

    







nIterations     = 1000
p               = 120
nPatterns       = p
i               = 4
nInputNodes     = i
h               = 8
nHiddenNodes    = h
j               = 3
nOutputNodes    = j

eta     	= 0.04
alpha           = 0.02


out1 = np.zeros([p,h])
out2 = np.zeros([p,j])
delta1 = np.zeros([p,h])
delta2 = np.zeros([p,j])
delw2 = np.zeros([j,h])
delw1 = np.zeros([h,i])



file_weights1 	= "weight1.csv"
file_weights2 	= "weight2.csv"

wg1     = pd.read_csv(file_weights1, header=None)
wg2     = pd.read_csv(file_weights2, header=None)

w1      = wg1.values
w2      = wg2.values

    

for xq in range (0,nIterations):

    # pattern ke - xp
    for xp in range(0,nPatterns):

        #hidden layer
        for h in range(0,nHiddenNodes):
            sums = 0
            for i in range(0,nInputNodes):
                a       =   w1[h][i]  *  out0[xp][i]
                sums    = sums + a
            out1[xp][h]   =   1.0  /  (1.0  +  np.exp(-sums))

        #output layer
        for j in range(0,nOutputNodes):
            sums = 0
            for h in range(0,nHiddenNodes):
                a       =   w2[j][h]  *  out1[xp][h]
                sums    = sums + a
            out2[xp][j]   =   1.0  /  (1.0  +  np.exp(-sums))

        #delta output
        for j in range(0,nOutputNodes):
            delta2[xp][j]   =   (target[xp][j] - out2[xp][j])  * out2[xp][j]   *   (1.0 - out2[xp][j]);


        #delta hidden
        for h in range(0,nHiddenNodes):
            sums = 0
            for j in range(0,nOutputNodes):
                a       =   delta2[xp][j] * w2[j][h]
                sums    = sums + a
            delta1[xp][h]  =  sums  *  out1[xp][h]  *  (1.0 - out1[xp][h]);


    for j in range(0, nOutputNodes):
        dw = 0
        sums = 0.0

        for p in range(0,nPatterns):
            a =  delta2[p][j]
            sums = sums + a

        dw   =   eta * sums  +  alpha * delw2[j][nHiddenNodes-1]
        w2[j][nHiddenNodes-1]   +=   dw
        delw2[j][nHiddenNodes-1] =   dw

        for h in range(0,nHiddenNodes):
          sums = 0.0;

          for p in range(0,nPatterns):
             a =  delta2[p][j] * out1[p][h];
             sums = sums + a

          dw           =   eta * sums  +  alpha * delw2[j][h];
          w2[j][h]     +=  dw;
          delw2[j][h]  =   dw;


    for h in range(0, nHiddenNodes):
        dw = 0
        sums = 0.0

        for p in range(0,nPatterns):
            a =  delta1[p][h]
            sums = sums + a

        dw   =   eta * sums  +  alpha * delw1[h][nInputNodes-1]
        w1[h][nInputNodes-1]   +=   dw
        delw1[h][nInputNodes-1] =   dw

        for i in range(0,nInputNodes):
          sums = 0.0;

          for p in range(0,nPatterns):
             a =  delta1[p][h] * out1[p][i];
             sums = sums + a

          dw           =   eta * sums  +  alpha * delw1[h][i];
          w1[h][i]     +=  dw;
          delw1[h][i]  =   dw;






df1 = pd.DataFrame(w1)
df2 = pd.DataFrame(w2)
df1.to_csv("weight1_new.csv", index=False, header=False)
df2.to_csv("weight2_new.csv", index=False, header=False)












