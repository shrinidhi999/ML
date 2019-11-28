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

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\Iris.csv',usecols=[1,2,3,4,5])
print(dataset.info())
# dataset.groupby('Species').size()
# dataset.hist()
# plt.show()


X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,4:5].values
y = y.reshape(-1,1)

# from sklearn.ensemble import ExtraTreesClassifier
# mod = ExtraTreesClassifier()
# mod.fit(X,y)
# print(mod.feature_importances_)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7, stratify=y)

df_ytrain = pd.DataFrame(y_train)
df_ytest = pd.DataFrame(y_test)

print(df_ytrain[0].value_counts())
print(df_ytest[0].value_counts())

# from sklearn.preprocessing import StandardScaler
# sc= StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# from sklearn.decomposition import PCA
# pca= PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# e_var = pca.explained_variance_ratio_
# e_var =e_var.reshape(-1,1)

# from sklearn.feature_selection import RFE
# mod = LogisticRegression()
# rfe = RFE(mod,2)
# fit = rfe.fit(X,y)
# print("Num Features: %s" % (fit.n_features_))
# print("Selected Features: %s" % (fit.support_))
# print("Feature Ranking: %s" % (fit.ranking_))

# models =[]
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
#

# for name,model in models:
#     kfold = model_selection.KFold(n_splits=10,random_state=7)
#     cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
# #    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)
# #     print(confusion_matrix(y_train, cv_predict))
#     print(f"{name}: {cv_res.mean()}: {cv_res.std()}")
#


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pres= knn.predict(X_test)
print(accuracy_score(y_test,pres))
print(confusion_matrix(y_test,pres))
print(classification_report(y_test,pres))


# knn = SVC(gamma='auto')
# knn.fit(X_train,y_train)
# pres= knn.predict(X_test)
# print(accuracy_score(y_test,pres))
# print(confusion_matrix(y_test,pres))
# print(classification_report(y_test,pres))
#

