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
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import math
from matplotlib import pyplot as plt

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\DL\ionosphere.csv')
dataset.scatter_matrix()

dataset.isnull().sum()

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')
dataset.info()

X = dataset.drop('label', axis=1)
y = dataset['label']


y.value_counts()
y = pd.get_dummies(y, drop_first=True)

sc =StandardScaler()
X = sc.fit_transform(X)

def step_decay(epoch):
    lr = 0.1
    epoch = 1 if epoch == 0 else epoch
    n_div = math.ceil(epoch / 10)
    return lr/n_div
    
model = Sequential()

model.add(Dense(34, kernel_initializer='normal', activation='relu', input_dim=34))
#model.add(Dropout(0.2))

model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

#sgd = SGD(lr = 0.1, momentum=0.8, decay = .1/50, nesterov=False)
#acc: 0.9915 - val_loss: 0.0631 - val_acc: 0.9914

sgd = SGD(lr = 0.0, momentum=0.8, decay = 0, nesterov=False)
#acc: 0.9915 - val_loss: 0.0780 - val_acc: 0.9828
callback_list = [LearningRateScheduler(step_decay)]

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, validation_split=0.33, batch_size=12, epochs=50, callbacks=callback_list)
k = history.history

plt.plot(k['loss'])
plt.plot(k['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['train','test'])
