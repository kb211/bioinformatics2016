import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from keras.models import Sequential
from keras.layers import Dense
import sklearn
from keras.callbacks import EarlyStopping
class MLPC:

    def __init__(self, loadModel=False):
        
        self.loadModel = loadModel
        if self.loadModel:
           from sklearn.externals import joblib
           path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/MLP.pkl'
           self.model = joblib.load(path)
        else:
            self.model = Sequential()
            self.model.add(Dense(400, input_dim=631, activation='relu'))
            #self.model.add(Dense(8))
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mse')
            
    def train(self, X, Y):
        self.X = X
        self.Y = Y
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(self.X,self.Y, verbose=2, nb_epoch=20, validation_split=0.2, callbacks=[early_stopping])
    
    def test(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=20)
        return score
        

    def predict(self, X):
        proba = self.model.predict_proba(X)
        Z = self.model.predict(X)
        return Z, proba       

    def save(self):
        from sklearn.externals import joblib
        path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/MLP.pkl'
        joblib.dump(self.model, path)
        
    def load(self):
        from sklearn.externals import joblib
        path = os.path.dirname(os.path.abspath(__file__)) + '/picklefiles/MLP.pkl'
        self.model = joblib.load(path)
        return self.model
        

