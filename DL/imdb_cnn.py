from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import pandas as pd
from keras.preprocessing import sequence
from keras.layers import Embedding

top_words=5000
max_words=500
(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=top_words)

seed =7
np.random.seed(seed)


X= np.concatenate((X_train, X_test), axis=0)
y= np.concatenate((y_train, y_test), axis=0)
#
#print(len(np.unique(X)))
#print(np.unique(y))

X_train =sequence.pad_sequences(X_train, maxlen=max_words)
X_test =sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=5)

print(model.evaluate(X_test, y_test))