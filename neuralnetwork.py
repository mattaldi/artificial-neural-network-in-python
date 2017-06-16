import numpy as np

"""

original code: Neural Network PC Tools by Russell C. Eberhart and Roy W. Dobbins

edited by: Muhamad Aldiansyah

This original code is written in C taken from: Neural Network PC Tools by Russell C. Eberhart and Roy W. Dobbins

The implementation from the book is using command line interface (cli). I edited the code so it can running in IDE like Eclipse for C, and i use iris data set for this classification task.

"""

nIterations = 500
p = 120
nPatterns = p
i = 4
nInputNodes  = i
h = 8
nHiddenNodes = h
j = 3
nOutputNodes = j

eta 	= 0.04
alpha = 0.02

file_input 		= "iris_datalatih.txt"
file_target 	= "iris_target_latih.txt"
file_weights1 	= "iris_bobot1.txt"
file_weights2 	= "iris_bobot2.txt"


out1 = np.zeros([p,h])
out2 = np.zeros([p,j])
delta1 = np.zeros([p,h])
delta2 = np.zeros([p,j])
delw2 = np.zeros([j,h])
delw1 = np.zeros([h,i])




def convarr(x):
    x = np.array(x)
    x = x.astype(np.float)
    return x

with open(file_input) as textFile:
    out0 = [line.split() for line in textFile]

with open(file_target) as textFile:
    target = [line.split() for line in textFile]

with open(file_weights1) as textFile:
    w1 = [line.split() for line in textFile]

with open(file_weights2) as textFile:
    w2 = [line.split() for line in textFile]

out0 = convarr(out0)
target = convarr(target)
w1 = convarr(w1)
w2 = convarr(w2)

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






##/* --------------- akhir latih -----------------------------*/
