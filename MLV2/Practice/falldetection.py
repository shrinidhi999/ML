#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\\falldetection.csv')

#Taking care of Missing Data in Dataset
print(dataset.isnull().sum())

#Duplicate check
#print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')


# Correlation check 
# using heat map - phi_k
import phik
from phik import resources, report

corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

corr = dataset.phik_matrix()['ACTIVITY'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.10]
dataset.drop(to_drop_1, axis=1, inplace=True)
#
corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.9:
            print(f"{col[i]} -{col[j]}")

#dataset.drop(['CIRCLUATION'],inplace=True,axis=1)


X = dataset.drop(['ACTIVITY'],axis=1)
y = dataset['ACTIVITY']

#Remove outliers - 
from scipy import stats
numeric_cols = X.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))

for i in range(numeric_cols.shape[0]):
    for j in range(numeric_cols.shape[1]):
        if z[i,j] >= 3:
#            print(f"{i} -{j}")
            numeric_cols.iloc[i,j] = numeric_cols.iloc[:,j].median()

X.update(numeric_cols)            


# Variance check - if variance less than threshold remove them
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0.5)
constant_filter.fit(X)
print(X.columns[constant_filter.get_support()])

X= X.values
y= y.values


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X=mm.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

#Splitting the Dataset into Training set and Test Set
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)

#Model Fit

# Classification
models =[]
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=100, max_depth=None)))
models.append(('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=100, max_depth=None)))

for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=7)
    cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
    print(f"{name}: {cv_res.mean()}: {cv_res.std()}")
#
##
###select  the model with good score.

#from sklearn.model_selection import GridSearchCV
#params = [{'n_estimators':[1,10,15,100], 'max_depth':[None,1,2,3,4]}]
#vc = GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,
#                  verbose=1,cv=10,n_jobs=-1)
#res = vc.fit(X_train,y_train)
#print(res.best_score_)
#print(res.best_params_)

mod = RandomForestClassifier(n_estimators=100, max_depth=None)
mod.fit(X_train, y_train)

y_pred = mod.predict(X_test)
y_pred = y_pred.reshape(-1,1)

from sklearn.metrics import accuracy_score, classification_report
print("##########      RF       #############")
print(f"accuracy_score : {accuracy_score(y_pred, y_test)}")
print(f"{classification_report(y_pred, y_test)}")


mod = ExtraTreesClassifier(n_estimators=125, n_jobs=-1)
mod.fit(X_train, y_train)

y_pred = mod.predict(X_test)
y_pred = y_pred.reshape(-1,1)

from sklearn.metrics import accuracy_score
print("##########   ET     #############")
print(f"accuracy_score : {accuracy_score(y_pred, y_test)}")
print(f"{classification_report(y_pred, y_test)}")
