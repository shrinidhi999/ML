import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Data.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,3:4].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
X[:,1:3] = imp.fit_transform(X[:,1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])
X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray()
y = LabelEncoder().fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

