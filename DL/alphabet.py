from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 

c_i = dict((c,i) for i,c in enumerate(alphabet))
i_c = dict((i,c) for i,c in enumerate(alphabet))

X_train = []
y_train = []

look_back = 3

def get_train_set(dataset):
    X_train = []
    y_train = []
    for i in range(look_back, len(dataset)):
        seq_in = dataset[i-look_back: i]
        seq_out = dataset[i]
        seq_in = [c_i[c] for c in seq_in]
        seq_out = c_i[seq_out]
        
        X_train.append(seq_in)
        y_train.append(seq_out)
    
    return X_train, y_train   


X_train, y_train = get_train_set(alphabet)
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))

X_train1=X_train
X_train = X_train/float(len(alphabet))
y_train = np_utils.to_categorical(y_train)

model = Sequential()
model.add(LSTM(30, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=1)

scores = model.evaluate(X_train, y_train)


for pattern in X_train1: 
    x = np.reshape(pattern, (1, len(pattern), 1)) 
    x = x / float(len(alphabet)) 
    prediction = model.predict(x, verbose=0) 
    index = np.argmax(prediction) 
    result = i_c[index]     
    print(result)