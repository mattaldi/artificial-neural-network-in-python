import numpy as np
import pandas as pd

fileCsv = 'iriscsv.csv'

datas   = pd.read_csv(fileCsv, header=None, parse_dates=True)
data    = datas.values

f1      = data[:,0]
f2      = data[:,1]
f3      = data[:,2]
f4      = data[:,3]
label   = data[:,4]
feature = data[:,0:4]

labelnum = []
for i in range(0,len(label)):
    labelnow = label[i]
    if (labelnow == "Iris-setosa"):
        labelnum = np.append(labelnum, 1)
    if (labelnow == "Iris-versicolor"):
        labelnum = np.append(labelnum, 2)
    if (labelnow == "Iris-virginica"):
        labelnum = np.append(labelnum, 3)

    
