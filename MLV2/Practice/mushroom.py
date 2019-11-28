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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

dataset = pd.read_csv('.\Datasets\mushrooms.csv')
print(dataset.groupby('class').size())
print(dataset.shape)

le = LabelEncoder()
for i in dataset.columns:
    dataset[i] = le.fit_transform(dataset[i])


dataset.head()

print(dataset.isnull().sum())

dataset.hist()
plt.show()


X = dataset.iloc[:,1:]


X = pd.get_dummies(X,columns=X.columns,drop_first=True).values
y = dataset.iloc[:,0:1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

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


knn = SVC(gamma='auto')
knn.fit(X_train,y_train)
pres= knn.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))



knn = DecisionTreeClassifier()
knn.fit(X_train,y_train)
pres= knn.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))


knn = LogisticRegression()
knn.fit(X_train,y_train)
pres= knn.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))

