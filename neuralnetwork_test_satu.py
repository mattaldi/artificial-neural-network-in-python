import numpy as np

out0  = np.array([5.7, 3, 4.2, 1.2])
label = ["setosa","versicolor","virginica"]
nIterations = 500
p = 120
nPatterns = p
i = 4
nInputNodes  = i
h = 8
nHiddenNodes = h
j = 3
nOutputNodes = j


def convarr(x):
    x = np.array(x)
    x = x.astype(np.float)
    return x

file_weights1 	= "iris_bobot1_new.txt"
file_weights2 	= "iris_bobot2_new.txt"

with open(file_weights1) as textFile:
    w1 = [line.split() for line in textFile]

with open(file_weights2) as textFile:
    w2 = [line.split() for line in textFile]

w1 = convarr(w1)
w2 = convarr(w2)
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
