import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Social_Network_Ads.csv')
X = ds.iloc[:,2:-1].values
y = ds.iloc[:,4:5].values
yz = ds.iloc[:,4:5]

from time import time
s=time()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
print(time()-s)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=nb, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)