import sklearn.ensemble as en
import os

class random_forrest:

    def __init__(self, loadModel=False):
        
        self.loadModel = loadModel
        if self.loadModel:
           from sklearn.externals import joblib
           path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/MLP.pkl'
           self.model = joblib.load(path)
        else:
#            self.model = en.RandomForestRegressor(oob_score=True, n_jobs=20, n_estimators=2000)
            self.model = en.RandomForestRegressor(n_estimators=100)
            
    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.model.fit(self.X,self.Y)
    
    def test(self, X, Y):
        score = self.model.score(X, Y)
        return score
        
    def predict(self, X):
        Z = self.model.predict(X)
        return Z

    def save(self):
        from sklearn.externals import joblib
        path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/BoostedRT.pkl'
        joblib.dump(self.model, path)
        
    def load(self):
        from sklearn.externals import joblib
        path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/BoostedRT.pkl'
        self.model = joblib.load(path)
        return self.model
        