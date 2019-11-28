import pandas as pd
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\Iris.csv')
X = dataset.drop(['Species','Id'], axis=1)
y = dataset['Species']
y = pd.get_dummies(y)

def build_model():
    classifier = Sequential()
    classifier.add(Dense(4, activation='relu', kernel_initializer='normal', input_dim = 4))
    classifier.add(Dense(3, activation='sigmoid', kernel_initializer='normal'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


k_c = KerasClassifier(build_fn= build_model, batch_size = 5, epochs=200)
k_f = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(k_c, X, y, cv=k_f)
print(results.mean())
print(results.std())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7, stratify=y)


classifier = Sequential()
classifier.add(Dense(4, activation='relu', kernel_initializer='normal', input_dim = 4))
classifier.add(Dense(3, activation='softmax', kernel_initializer='normal'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X, y, batch_size = 5, epochs=200)

def update(x):
    if x >= 0.5:      
        return 1
    else:
        return 0
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)
for col in y_pred.columns:
    y_pred[col] = y_pred[col].apply(lambda x: update(x))
    
    
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))    