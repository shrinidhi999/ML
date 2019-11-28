import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Wine.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,13:14].values


from sklearn import model_selection
X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=7)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train =pca.fit_transform(X_train)
X_test=pca.transform(X_test)
# e_var = pca.explained_variance_ratio_
# e_var = e_var.reshape(-1,1)

from sklearn import naive_bayes
lr = naive_bayes.GaussianNB()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))