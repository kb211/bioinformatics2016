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

    rf = readFile()
    X, Y, gene_labels = rf.splitTarget()
    
    #normailizing data
    X_norm = normalize(X)
  
    print "Cross-validating genes..."
    score_Tree, mse_Tree = setup_x_validation("BoostedRT", X, Y, gene_labels)
    score_nn, mse_nn = setup_x_validation("neuralnet", X_norm, Y, gene_labels)
    
    print "Spearman correlation Boosted trees: ", score_Tree
    print "Spearman correlation Neural network: ", score_nn


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

def setup_x_validation(model_type, X, Y, gene_labels):
    n_folds=17
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(gene_labels)
    gene_classes = label_encoder.transform(gene_labels)
    cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds, shuffle=True)
    #fold_labels = ["fold%d" % i for i in range(1,n_folds+1)]
    cv = [c for c in cv] 

    scores = []
    mse_scores = []
    
    model = [None]*len(cv)
    
    for i,fold in enumerate(cv):
        train,test = fold
        print "\n working on fold %d of %d, with %d train and %d test" % (i, len(cv), len(train), len(test))
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for index in train:
            x_train.append(X[index])
            y_train.append(Y[index])
        if model_type == "neuralnet":
            model[i] = MLPC()
        elif model_type == "BoostedRT":
            model[i] = BoostedRT()
        else:
            print "No model specified"
            break
        
        model[i].train(x_train, y_train)
        
        for index2 in test:
            x_test.append(X[index2])
            y_test.append(Y[index2])
 
        Z = model[i].predict(x_test)
        r, p = util.spearmanr_nonan(Z , y_test)
        scores.append(r)
        
        mse_score = model[i].test(x_test, y_test)
        mse_scores.append(mse_score)
        
    final_mse = np.array(mse_scores).sum()/len(mse_scores)
    finalscore = np.array(scores).sum()/len(scores)
    return finalscore, final_mse
    
if __name__ == "__main__":
    main(sys.argv[1:])


