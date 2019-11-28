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


dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\titanic\train.csv')

X = dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket','Cabin'], axis=1)
y = dataset['Survived']

X.isnull().sum()
X['Age'] =  X['Age'].fillna(X['Age'].median())
X['Embarked'] = X['Embarked'].fillna('S')

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)
y_test.value_counts()


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression()
#lr.fit(X_train, y_train)
#
#y_pred = lr.predict(X_test)
#from sklearn.metrics import accuracy_score
#accuracy_score(y_pred, y_test)

from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(5, input_shape=(8,), activation='relu', kernel_initializer='uniform'))                 
classifier.add(Dropout(rate=0.1))
 
classifier.add(Dense(5, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))
 
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=2, epochs=200)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)


# Grid search
def build_model(init='uniform', optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(8, input_shape=(8,), activation='relu', kernel_initializer='normal'))                 
    classifier.add(Dropout(rate=0.1))
     
    classifier.add(Dense(4, activation='relu', kernel_initializer='normal'))
    classifier.add(Dropout(rate=0.1))
     
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def grid_search():
    params = {
            "init" : ['uniform', 'normal', 'glorot_uniform']
            ,"optimizer" : ['adam', 'rmsprop']
            ,"epochs": [50,100,200]
            ,"batch_size": [5,10,2]
            }
    
    k_c = KerasClassifier(build_fn = build_model)
    g_s = GridSearchCV(estimator=k_c, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)    
    g_s = g_s.fit(X_train, y_train)
    print(g_s.best_params_)
    print(g_s.best_score_)

grid_search()

# {'batch_size': 10, 'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop'}
# 0.8286516853932584

from keras.optimizers import SGD

classifier = Sequential()
classifier.add(Dense(8, input_shape=(8,), activation='relu', kernel_initializer='normal'))                 
classifier.add(Dropout(rate=0.1))
 
classifier.add(Dense(4, activation='relu', kernel_initializer='normal'))
classifier.add(Dropout(rate=0.1))
 
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
#adam - acc: 0.8428 - val_loss: 0.4635 - val_acc: 0.7957
sgd =SGD(lr=0.1, momentum=0.8, decay=0.1/200, nesterov=False)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=28, epochs=200, validation_split=0.33)


