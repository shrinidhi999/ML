import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Social_Network_Ads.csv')
X = ds.iloc[:,2:-1].values
y = ds.iloc[:,4:5].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# from sklearn.svm import SVC
# rf = SVC(kernel = 'rbf', random_state=0)
# rf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)