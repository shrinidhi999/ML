import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\DL\Passengers.csv', skipfooter=3, engine='python')
dataset = dataset.iloc[:,1:2]

lg = int(len(dataset)*.67)
train_set = dataset.iloc[0: lg, :]
test_set = dataset.iloc[lg:, :]

sc = MinMaxScaler()
scaled_train_set = sc.fit_transform(train_set)

#scaled_test_set = sc.transform(test_set)



look_back = 20

def get_train_set(dataset):
    X_train = []
    y_train = []
    for i in range(look_back, len(dataset)):
        X_train.append(dataset[i-look_back: i, 0])
        y_train.append(dataset[i, 0])
    
    return X_train, y_train    

X_train, y_train = get_train_set(scaled_train_set)

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=70, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=70))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=32, epochs=100)


test_inputs = train_set.iloc[len(train_set)-look_back:len(train_set), 0]
test_inputs = pd.concat((test_inputs, test_set.iloc[:,0]), axis=0, ignore_index=True).values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = sc.transform(test_inputs)

X_test, _ = get_train_set(test_inputs)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred = model.predict(X_test)
pred = sc.inverse_transform(pred)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(test_set, pred)))

test_set = test_set.reset_index(drop=True)

plt.plot(test_set, color='red')
plt.plot(pred, color='blue')
plt.xlabel('Time')
plt.ylabel('Passengers')
#plt.legend()
plt.show()