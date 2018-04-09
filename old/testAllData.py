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
out0test    = fdtest



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
        target[x,1] = 1
    if (labelnow == "Iris-versicolor"):
        target[x,1] = 1
    if (labelnow == "Iris-virginica"):
        target[x,1] = 1
    x = x + 1

    







nIterations     = 1000
p               = 30
nPatterns       = p
i               = 4
nInputNodes     = i
h               = 8
nHiddenNodes    = h
j               = 3
nOutputNodes    = j

eta     	= 0.04
alpha           = 0.02
nTest           = 30

out1 = np.zeros([p,h])
out2 = np.zeros([p,j])
delta1 = np.zeros([p,h])
delta2 = np.zeros([p,j])
delw2 = np.zeros([j,h])
delw1 = np.zeros([h,i])


hasil = []
file_weights1 	= "weight1_new.csv"
file_weights2 	= "weight2_new.csv"

wg1     = pd.read_csv(file_weights1, header=None)
wg2     = pd.read_csv(file_weights2, header=None)

w1      = wg1.values
w2      = wg2.values

label = ["setosa","versicolor","virginica"]



# pattern ke - xp
for xp in range(0,nTest):

    #hidden layer
    for h in range(0,nHiddenNodes):
        sums = 0
        for i in range(0,nInputNodes):
            a       =   w1[h][i]  *  out0test[xp][i]
            sums    = sums + a
        out1[xp][h]   =   1.0  /  (1.0  +  np.exp(-sums))

    #output layer
    for j in range(0,nOutputNodes):
        sums = 0
        for h in range(0,nHiddenNodes):
            a       =   w2[j][h]  *  out1[xp][h]
            sums    = sums + a
        out2[xp][j] =   1.0  /  (1.0  +  np.exp(-sums))


    zz = out2[xp][:]
    mval = max(zz)
    i = zz.tolist().index(mval)
    vv = label[i]
    hasil = np.append(hasil,vv)



print hasil


