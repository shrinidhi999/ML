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
from matplotlib import pyplot as plt
from keras.layers import Dropout

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\DL\diabetes.csv')
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

def history():
    sc = StandardScaler()
    X1 = X
    X1 = sc.fit_transform(X1)
    classifier = Sequential()
    classifier.add(Dense(10, input_dim=8, activation='relu', kernel_initializer='normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(5, activation='relu', kernel_initializer='normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    hist = classifier.fit(X1, y, batch_size=2, epochs=200, validation_split=0.33)
    
    arr = np.array(X.loc[0]).reshape(1,-1)
    y_pred_new = classifier.predict(arr)
    return hist
    
h = history()    

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.ylabel='Accuracy'
plt.xlabel='epochs'
plt.title='Accuracy'
plt.legend(['train', 'test'], loc='upper left')


plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.ylabel='loss'
plt.xlabel='epochs'
plt.title='loss'
plt.legend(['train', 'test'], loc='upper left')
    
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)
y_test.value_counts()

#Check SMOTE

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def normal_call():
    classifier = Sequential()
    
    classifier.add(Dense(5, input_shape=(8,), activation='relu', kernel_initializer='uniform'))                 
    classifier.add(Dense(5, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    classifier.fit(X_train, y_train, batch_size=10, epochs=150)
    
    y_pred = classifier.predict(X_test)
    y_pred = y_pred > 0.5
    
    from sklearn.metrics import accuracy_score
    accuracy_score(y_pred, y_test)
    accuracy_score(y_test, y_pred)
    
    ss = classifier.evaluate(X_test, y_test)

#normal_call()

# Cross val score
def build_model():
    classifier = Sequential()
    classifier.add(Dense(5, input_shape=(8,), activation='relu', kernel_initializer='uniform'))                 
    classifier.add(Dense(5, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def kfold_score():
    k_c = KerasClassifier(build_fn= build_model, batch_size=10, epochs=150)
    s_k = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    scores = cross_val_score(estimator=k_c, X=X_train, y=y_train, n_jobs=-1, cv=s_k)
    print(scores.mean())
    print(scores.std())
    
#kfold_score()

# Grid search
def build_model(init='uniform', optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(5, input_shape=(8,), activation='relu', kernel_initializer=init))                 
    classifier.add(Dense(5, activation='relu', kernel_initializer=init))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def grid_search():
    params = {
            "init" : ['uniform', 'normal', 'glorot_uniform']
            ,"optimizer" : ['adam', 'rmsprop']
            ,"epochs": [50,100,200]
            ,"batch_size": [5,10,20]
            }
    
    k_c = KerasClassifier(build_fn = build_model)
    g_s = GridSearchCV(estimator=k_c, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)    
    g_s = g_s.fit(X_train, y_train)
    print(g_s.best_params_)
    print(g_s.best_score_)

grid_search()

# Tuned model
classifier = Sequential()

classifier.add(Dense(5, input_shape=(8,), activation='relu', kernel_initializer='uniform'))                 
classifier.add(Dense(5, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=20, epochs=50)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)
accuracy_score(y_test, y_pred)

ss = classifier.evaluate(X_test, y_test)

