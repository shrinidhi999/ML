import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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

dataset = pd.read_csv('D:\ML\ML_Rev\\titanic\\train.csv', usecols=[1,2,4,5,11])
dataset =dataset.dropna()
print(dataset.shape)
print(dataset.info())
dataset.groupby('Survived').size()
print(dataset.describe())
print(dataset.isnull().sum())


# sex_pivot = dataset.pivot_table(index = 'Sex', values='Survived')
# sex_pivot.plot.bar()
# plt.show()
#
# Pclass_pivot = dataset.pivot_table(index = 'Pclass', values='Survived')
# Pclass_pivot
#
# embark_pivot = dataset.pivot_table(index="Embarked",values="Survived")
# embark_pivot


dataset['Sex'][dataset['Sex'] == 'female'] = 1
dataset['Sex'][dataset['Sex'] == 'male'] = 0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset['Pclass'] = le.fit_transform(dataset['Pclass'])
dataset= pd.get_dummies(dataset,columns=['Pclass'],drop_first=True)

dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
dataset= pd.get_dummies(dataset,columns=['Embarked'],drop_first=True)

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,1:2].values
y = y.astype('int')

from sklearn import preprocessing
ss = preprocessing.StandardScaler()
X = ss.fit_transform(X)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)

models =[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=7)
    cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
#    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)
#     print(confusion_matrix(y_train, cv_predict))
    print(f"{name}: {cv_res.mean()}: {cv_res.std()}")


lr = LogisticRegression()
lr.fit(X_train,y_train)
pres= lr.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))


svc = SVC(gamma='auto')
svc.fit(X_train,y_train)
pres= svc.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))


# Test set

X_dataset = pd.read_csv('.\Datasets\\titanic\\test.csv', usecols=[1,3,4,10])
X_dataset = X_dataset.fillna(X_dataset.mean())

X_dataset['Sex'][X_dataset['Sex'] == 'female'] = 1
X_dataset['Sex'][X_dataset['Sex'] == 'male'] = 0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_dataset['Pclass'] = le.fit_transform(X_dataset['Pclass'])
X_dataset= pd.get_dummies(X_dataset,columns=['Pclass'],drop_first=True)

X_dataset['Embarked'] = le.fit_transform(X_dataset['Embarked'])
X_dataset= pd.get_dummies(X_dataset,columns=['Embarked'],drop_first=True)

print(X_dataset.isnull().sum())

X = X_dataset.values


y=pd.read_csv('.\Datasets\\titanic\\gender_submission.csv', usecols=[1]).values
y = y.astype('int')

from sklearn import preprocessing
ss = preprocessing.StandardScaler()
X = ss.fit_transform(X)

# print(len(X))
# print(len(pres))
# print(len(y))

pres= lr.predict(X)
print(accuracy_score(y,pres))
print(confusion_matrix(y,pres))
print(classification_report(y,pres))


pres= svc.predict(X)
print(accuracy_score(y,pres))
print(confusion_matrix(y,pres))
print(classification_report(y,pres))