import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\DL\Churn_Modelling.csv')

X = dataset.iloc[:,3:-1]
y = dataset.iloc[:,-1]

X = pd.get_dummies(X, drop_first=True)

y.value_counts()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)
y_test.value_counts()

#Check SMOTE

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(6, input_shape=(11,), activation='relu', kernel_initializer='uniform'))                 
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=100, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


X_new = sc.fit_transform(np.array([[600, 40, 3, 60000, 2, 1, 1, 50000, 0,0,1]]))
y_pred1 = classifier.predict(X_new)
y_pred1 = y_pred1>0.5


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

keras_classifier = KerasClassifier(build_fn= build_classifier, batch_size=100, epochs=100)

accuracies = cross_val_score(estimator=keras_classifier, X = X_train, y = y_train, cv=10, n_jobs=-1)
print(accuracies.mean())
print(accuracies.std())


### PARAM tunig #########3

from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

keras_classifier = KerasClassifier(build_fn= build_classifier)
params = { 
        'batch_size':[25,32,100],
        'epochs':[100,200,500],
        'optimizer':['adam','rmsprop']
        }

g_s = GridSearchCV(estimator=keras_classifier, param_grid=params, scoring='accuracy', cv=10)    
g_s = g_s.fit(X_train, y_train)
print(g_s.best_params_)
print(g_s.best_score_)