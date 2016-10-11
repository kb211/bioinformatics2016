from readFile import readFile
from MLP import MLPC
import sys
import json
import math
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc
import sklearn.metrics
import sklearn.cross_validation
import multiprocessing

def main(argv):
    
    rf = readFile()
    X, Y, gene_labels = rf.splitTarget()
    
    #normailizing data
    #X = normalize(X)
  

    print "Cross-validating genes..."
    setup_x_validation(X, Y, gene_labels)


def normalize(data):

    data = np.array(data).astype(float)
    minimum = np.array(data).min(axis=0)
    maximum = np.array(data).max(axis=0)
    data = (data - minimum)/maximum
    data = data.tolist()
    return data

def setup_x_validation(X, Y, gene_labels):
    n_folds=17
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(gene_labels)
    gene_classes = label_encoder.transform(gene_labels)
    cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds, shuffle=True)
    #fold_labels = ["fold%d" % i for i in range(1,n_folds+1)]
    cv = [c for c in cv] 

    jobs = []

    for i,fold in enumerate(cv):
        train,test = fold
        print "working on fold %d of %d, with %d train and %d test" % (i, len(cv), len(train), len(test))
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for index in train:
            x_train.append(X[index])
            y_train.append(Y[index])
        MLP = MLPC()
        MLP.train(x_train, y_train)
        
        for index in test:
            x_test.append(X[index])
            y_test.append(Y[index])
        Z = MLP.predict(x_test, y_test)
        
        jobs.append(MLP)
        
    
if __name__ == "__main__":
    main(sys.argv[1:])


