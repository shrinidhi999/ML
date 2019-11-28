import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from sklearn.pipeline import Pipeline


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\sonar.csv', header=None)

X = dataset.iloc[:,0:60]
y = dataset.iloc[:, -1]

y['R'].value_counts()

seed =7
np.random.seed(seed)

y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Normal
def normal():
    classifier = Sequential()
    
    classifier.add(Dense(60, input_shape=(60,), activation='relu', kernel_initializer='uniform'))                 
     
    classifier.add(Dense(30, activation='relu', kernel_initializer='uniform'))
    
    classifier.add(Dense(15, activation='relu', kernel_initializer='uniform'))
    
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    classifier.fit(X_train, y_train, batch_size=2, epochs=100)
    
    y_pred = classifier.predict(X_test)
    y_pred = y_pred > 0.7
    from sklearn.metrics import accuracy_score
    accuracy_score(y_pred, y_test)



X = dataset.iloc[:,0:60]
y = dataset.iloc[:, -1]

y = pd.get_dummies(y, drop_first=True)

# K Fold
def build_model():
    classifier = Sequential()
    
    classifier.add(Dense(60, input_dim=60, activation='relu', kernel_initializer='uniform'))
    
    classifier.add(Dense(30, input_dim=60, activation='relu', kernel_initializer='uniform'))
    
    
    classifier.add(Dense(1, input_dim=60, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return classifier

def kfold():
        
    estimator=[]
    estimator.append(('sc', StandardScaler()))
    estimator.append(('kc', KerasClassifier(build_fn=build_model, batch_size=2, epochs=100)))
    pl = Pipeline(estimator)
    
    
    s_k =StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    
    acc = cross_val_score(pl, X, y, cv=s_k)
    
    print(acc.mean())
    print(acc.std())
    

kfold()    
