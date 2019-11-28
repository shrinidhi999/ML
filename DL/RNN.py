import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\DL\Google_Stock_Price_Train.csv')
train_set = dataset.iloc[:, 1:2]


sc = MinMaxScaler()
scaled_train_set = sc.fit_transform(train_set)

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(scaled_train_set[i-60: i, 0])
    y_train.append(scaled_train_set[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, batch_size=32, epochs=100)


dataset_test = pd.read_csv(r'D:\ML\ML_Rev\Datasets\DL\Google_Stock_Price_Test.csv')
realtest_set = dataset_test.iloc[:, 1:2]


#dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis=0, ignore_index= True)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]

inputs = dataset.iloc[len(dataset)-60:len(dataset), 1]
inputs = pd.concat((inputs, dataset_test['Open']), axis=0, ignore_index= True).values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test =[]
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))    

pred = regressor.predict(x_test)
pred = sc.inverse_transform(pred)

plt.plot(realtest_set, color='red')
plt.plot(pred, color='blue')
plt.xlabel('Time')
plt.ylabel('Prices')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(realtest_set, pred))