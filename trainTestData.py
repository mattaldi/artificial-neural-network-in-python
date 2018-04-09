import numpy as np
import pandas as pd


nData               = 150
nCategory           = 3
nPercentTrainData   = 80
nFeature            = 4
nIterations         = 50
eta     	    = 0.04
alpha               = 0.02
hiddenNodes         = 5
fileCsv             = 'iriscsv.csv'
ofile_weights1 	    = "weight1.csv"
ofile_weights2 	    = "weight2.csv"

nfile_weights1 	    = "weight1_new.csv"
nfile_weights2 	    = "weight2_new.csv"



nPercentTrainData   = nPercentTrainData / float(100)
nPart       = nData/nCategory
nTrainData  = int(round(nPart * nPercentTrainData))
nTestData   = nPart - nTrainData

nTrainDataAll   = nTrainData * nCategory
nTestDataAll    = nTestData * nCategory
p               = nTrainDataAll
nPatterns       = p
i               = nFeature
nInputNodes     = i
h               = hiddenNodes
nHiddenNodes    = h
j               = nCategory
nOutputNodes    = j

out1            = np.zeros([p,h])
out2            = np.zeros([p,j])
delta1          = np.zeros([p,h])
delta2          = np.zeros([p,j])
delw2           = np.zeros([j,h])
delw1           = np.zeros([h,i])



datas   = pd.read_csv(fileCsv, header=None, parse_dates=True)
data    = datas.values

dataCollect = np.zeros(nCategory, dtype=object)

for i in range(0, nCategory):
    fd1 = nPart*i
    fd2 = nPart*i+nTrainData
    dataCollect[i]   = np.array(data[fd1:fd2,:])

dtrain  = dataCollect[0]
for i in range(1, nCategory):
    dtrain  = np.concatenate((dtrain,dataCollect[i]),axis=0)

fdtrain = dtrain[:,0:nFeature]
ldtrain = dtrain[:,nFeature]
out0    = fdtrain


hasil = []
data4   = data[40:50,:]
data5   = data[90:100,:]
data6   = data[140:150,:]
dtest   = np.concatenate((data4, data5, data6),axis=0)
fdtest  = dtest[:,0:4]
ldtest  = dtest[:,4]
out0test    = fdtest






x = 0
target = np.zeros([nTrainDataAll,nCategory])
for m in range(0,nCategory):
    for i in range(0,nTrainData):
        target[x,m] = 1
        x = x + 1


wg1     = pd.read_csv(ofile_weights1, header=None)
wg2     = pd.read_csv(ofile_weights2, header=None)

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





testTargetReal = np.zeros([nTestDataAll,nCategory])
testTargetObserv = np.zeros([nTestDataAll,nCategory])


x = 0
target = np.zeros([nTestDataAll,nCategory])
for m in range(0,nCategory):
    for i in range(0,nTestData):
        testTargetReal[x,m] = 1
        x = x + 1


# pattern ke - xp
for xp in range(0,nTestDataAll):

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
    testTargetObserv[xp][i] = 1





trueList = []
for xp in range(0,nTestDataAll):
    if ((testTargetObserv[xp] == testTargetReal[xp]).all()==True):
        trueList = np.append(trueList, 1)
    else:
        trueList = np.append(trueList, 0)
    

percentTrue = (sum(trueList)/nTestDataAll)*100
print percentTrue



df1 = pd.DataFrame(w1)
df2 = pd.DataFrame(w2)
df1.to_csv(nfile_weights1, index=False, header=False)
df2.to_csv(nfile_weights2, index=False, header=False)

