from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import pandas as pd


(X_train, y_train), (X_test, y_test)=mnist.load_data()

seed =7
np.random.seed(seed)


y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)
#y2 = pd.get_dummies(y_train, drop_first=True)

plt.subplot(221)
plt.imshow(X_test[1])

X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32')

X_train = X_train/255
X_test = X_test/255

def baseline_model():
    model = Sequential()
    model.add(Conv2D(30,(5,5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
#    model.add(Conv2D(15,(3,3), activation='relu'))
#    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
#    model.add(Dense(50, activation='relu'))
    model.add(Dense(y_train1.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = baseline_model()

model.fit(X_train, y_train1, epochs=10, batch_size=200, validation_split=0.33)

pred = model.predict(X_test)
pred = pred > 0.5
model.evaluate(X_test, y_test1)

p1 = pred.argmax(axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test1, pred))
print(confusion_matrix(y_test, p1))
print(classification_report(y_test, p1))