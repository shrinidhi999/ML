import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout



dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\housing.csv')

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

print((dataset.isnull().sum()/ len(dataset)) * 100)

for c in dataset.columns:
    dataset[c].fillna(dataset[c].median(), inplace=True)

X= dataset.drop('MEDV', axis=1)
y=dataset['MEDV']

    
np.random.seed(7)

def build_model():
    regr = Sequential()
    regr.add(Dense(20, input_dim=13, kernel_initializer='uniform', activation='relu'))
    regr.add(Dense(1, kernel_initializer='uniform'))
    regr.compile(optimizer='adam', loss='mean_squared_error')
    return regr

pl = Pipeline([('sc', StandardScaler()), ('reg', KerasRegressor(build_fn=build_model, epochs=100, batch_size = 2))])

kf = KFold(n_splits=10,shuffle=True, random_state=7)
acc = cross_val_score(pl, X,y,cv=kf, n_jobs=-1)
print(acc.mean())
print(acc.std())

# 13.761326057003117
# 7.320019195660228