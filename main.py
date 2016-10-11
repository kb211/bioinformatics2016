from readFile import readFile
import sys
from MLP import MLPC
from BoostedRT import BoostedRT
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.cross_validation
import util

def main(argv):
    MLP = MLPC()
    tree = BoostedRT()
    rf = readFile()
    X, Y, gene_labels = rf.splitTarget()
    
    #normailizing data
    X_norm = normalize(X)
  
    print "Cross-validating genes..."
    score_Tree = setup_x_validation(tree, X, Y, gene_labels)
    score_nn = setup_x_validation(MLP, X_norm, Y, gene_labels)
    print score_Tree, score_nn

def normalize(data):

    data = np.array(data).astype(float)
    minimum = data.min(0)
    maximum = data.ptp(0)
    
    for i in range(0,data.shape[0]):
        for j in range(0, data.shape[1]):
            if maximum[j] != 0:
                data[i,j] = (data[i,j] - minimum[j])/maximum[j]        

    data = data.tolist()
    return data

def setup_x_validation(model, X, Y, gene_labels):
    n_folds=17
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(gene_labels)
    gene_classes = label_encoder.transform(gene_labels)
    cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds, shuffle=True)
    #fold_labels = ["fold%d" % i for i in range(1,n_folds+1)]
    cv = [c for c in cv] 

    scores = []
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
            
        model.train(x_train, y_train)
        
        for index in test:
            x_test.append(X[index])
            y_test.append(Y[index])
            
        Z = model.predict(x_test)
        r, p = util.spearmanr_nonan(Z , y_test)
        scores.append(r)
    finalscore = np.array(scores).sum()/len(scores)
    return finalscore
    
if __name__ == "__main__":
    main(sys.argv[1:])


