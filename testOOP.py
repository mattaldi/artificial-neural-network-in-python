from neuralnet import *



nData               = 150
nCategory           = 3
nPercentTrainData   = 80
nFeature            = 4
nIterations         = 10
eta     	    = 0.04
alpha               = 0.02
hiddenNodes         = 8
fileCsv             = 'iriscsv.csv'
ofile_weights1 	    = "weight1.csv"
ofile_weights2 	    = "weight2.csv"
nfile_weights1 	    = "weight1_new.csv"
nfile_weights2 	    = "weight2_new.csv"

testnn  = neuralnet(nData, nCategory, nPercentTrainData,nFeature, nIterations, eta, alpha, hiddenNodes, fileCsv,ofile_weights1, ofile_weights2, nfile_weights1, nfile_weights2)

testnn.trainData()
