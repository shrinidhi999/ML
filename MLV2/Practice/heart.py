import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\heart.csv')

import pandas_profiling
profile = pandas_profiling.ProfileReport(dataset)
print(profile)
#print(dataset.info())
#print(dataset.isnull().sum())

print(dataset.describe())

print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(dataset.corr(method='spearman'), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
p

corr = dataset.corr(method='spearman')['target'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.2]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.corr(method='spearman').abs()
to_drop_2 = [col for col in corr.index if any((corr[col] > 0.5)&(corr[col] < 1))]
dataset.drop(['slope'], axis=1, inplace=True)

X = dataset.iloc[:,0:-1].values
y= dataset.target.values

#from sklearn.preprocessing import StandardScaler
#sc= StandardScaler()
#X = sc.fit_transform(X)

from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=0)


#from sklearn.decomposition import PCA
#pca= PCA(n_components=2)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#e_var = pca.explained_variance_ratio_
#e_var =e_var.reshape(-1,1)
#
#from sklearn.feature_selection import RFE
#mod = LogisticRegression()
#rfe = RFE(mod,2)
#fit = rfe.fit(X,y)
#print("Num Features: %s" % (fit.n_features_))
#print("Selected Features: %s" % (fit.support_))
#print("Feature Ranking: %s" % (fit.ranking_))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

models =[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


for name,model in models:
    cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)
    print(confusion_matrix(y_train, cv_predict))
    print(f"Accuracy score : {accuracy_score(y_train, cv_predict)}")
    print(f"{name}: {cv_res.mean()}: {cv_res.std()}")


svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(f"Accuracy score : {accuracy_score(y_test, y_pred)}")
print(f"classification_report : {classification_report(y_test, y_pred)}")
print(f"confusion_matrix : {confusion_matrix(y_test, y_pred)}")

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f"Accuracy score : {accuracy_score(y_test, y_pred)}")
print(f"classification_report : {classification_report(y_test, y_pred)}")
print(f"confusion_matrix : {confusion_matrix(y_test, y_pred)}")

