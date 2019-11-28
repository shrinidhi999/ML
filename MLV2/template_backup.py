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

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)
#
#import warnings
#warnings.filterwarnings('ignore')
#%reset -f
dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\cereal.csv')

# Display propertice
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
dataset.describe()

#Pandas profile
import pandas_profiling
profile = pandas_profiling.ProfileReport(dataset)
print(profile)

#Taking care of Missing Data in Dataset
print(dataset.columns[dataset.isnull().any()])
print(dataset.isnull().sum())
dataset['col'] = dataset['col'].fillna(dataset['col'].median())

#Filling missing cat values with high freq value
c_df = dataset.select_dtypes(['object','category'])

for col in c_df.columns:
    t = c_df[col].value_counts()
    c_df[col].fillna(t[t==t.max()].index[0],inplace=True)

#OR
    
for col in c_df.columns:
    c_df[col].fillna(c_df[col].mode(),inplace=True)
    
dataset.update(c_df)

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')


# box and whisker plots 
numeric_cols = dataset.select_dtypes(include=['int64','float64'])
numeric_cols.plot(kind='box', subplots=True, layout=(13,13), sharex=False, sharey=False) 
plt.show()

# histograms 
numeric_cols.hist() 
plt.show()


# Correlation check 
# using heat map - phi_k
import phik
from phik import resources, report

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


corr = corr['rating'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.2]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


# corr_mat = dataset.drop('rating',axis=1).corr(method='spearman').abs()
# to_drop = [col for col in corr_mat.columns if any((corr_mat[col] > 0.7)&(corr_mat[col] < 1))]

col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.8:
            print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")

dataset.drop(['potass','weight'],inplace=True,axis=1)


#Remove outliers -  try remove outliers using np.log
from scipy import stats
numeric_cols = dataset.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))
to_drop_rows=[]

##method 1
for i in range(numeric_cols.shape[0]):
    for j in range(numeric_cols.shape[1]):
        if z[i,j] >= 3:
            print(f"{i} -{j}")
            to_drop_rows.append(i)
            numeric_cols.iloc[i,j] = numeric_cols.iloc[:,j].median() #or mean()

##method 2
for j in range(numeric_cols.shape[1]):
    median = numeric_cols.iloc[:,j].median()
    for i in range(numeric_cols.shape[0]):        
        if z[i,j] >= 3:
            print(f"{i} -{j}")
            numeric_cols.iloc[i,j] = median #or mean()
# drop or replace by mean
#dataset = dataset.drop([to_drop_rows], axis=0)
dataset.update(numeric_cols)            

#For Categorical vars - remove/replace low freq vars

for col in dataset.select_dtypes(include=['category','object']).columns:
    dataset.loc[dataset[col].value_counts()[dataset[col]].values < 10, col] = np.nan

dataset['mfr'].value_counts()[dataset['mfr']].values < 10


## To replace categorical values with less than 10 freq

cnt = df['col'].value_counts()
cnt_indx = cnt[cnt < 10].index
df.loc[df['col'].isin(cnt_indx), 'col'] = 'others'




#Encoding categorical data
#drop_first - to avoid dummy variable trap
dataset = pd.get_dummies(dataset, drop_first=True)

# Variance check - if variance less than threshold remove them
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0.5)
constant_filter.fit(dataset)
print(dataset.columns[constant_filter.get_support()])
constant_columns = [column for column in dataset.columns if column not in dataset.columns[constant_filter.get_support()]]
dataset.drop(labels=constant_columns, axis=1, inplace=True)

#
#        Correlation check 
#                    using heat map - phi_k
#                    using chi square - Normalization: MinMaxScaler
#                    RFECV
#                     SelectFromModel- (Normalization: depends on model being used), using Extratreeclassifier/regressor or RFC/RFR
#         PCA/LDA - When categorical cols hav high cardinality
#


#The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to prove normal univariate distribution 

## skewness along the index axis 
#Skewness is a measure of the symmetry in a distribution. 
#A symmetrical dataset will have a skewness equal to 0. 
#So, a normal distribution will have a skewness of 0. 
#Skewness essentially measures the relative size of the two tails.
#skewed data affects intercept and coefficients when the model is fit
dataset.skew(axis = 0, skipna = True)
###Refer kc_house 2 

# find the kurtosis over the index axis 
#Kurtosis is all about the tails of the distribution — not the peakedness or flatness. 
#It is actually the measure of outliers present in the distribution.
#If there is a high kurtosis, then, we need to investigate why do we have so many outliers.
#If we get low kurtosis(too good to be true), then also we need to investigate and trim the dataset of unwanted results
dataset.kurt(axis = 0) 

#To remove skewness use following function
dataset[col] = np.cbrt(dataset[col]) # use this
dataset[col] = np.log(dataset[col]) #gives exceptions while tranforming better consider cube root


X= dataset[:,1:2]
y= dataset[:,-1]
#Splitting the Dataset into Training set and Test Set
#stratify parameter ll divide the dataset such thtat- y valyes are proprtionately divided among the test and train sets
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7, stratify=y)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca= PCA(n_components=None)
x = pca.fit_transform(x)
e_var = pca.explained_variance_ratio_
e_var =e_var.reshape(-1,1)

#Feature selection using RFECV
#Apply RFECV on sample set if dataset is too big
#Optimal no. of features given by RFECV may change with each run--> BE CAREFUL
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestRegressor() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='neg_mean_squared_error')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]

from sklearn.pipeline import Pipeline
# create pipeline 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('lr', LogisticRegression())) 
model1 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('lda', LinearDiscriminantAnalysis())) 
model2 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('knn', KNeighborsClassifier())) 
model3 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('cart', DecisionTreeClassifier())) 
model4 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('nb', GaussianNB())) 
model5 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('svc', SVC())) 
model6 = Pipeline(estimators) 

models =[]
models.append('LR', model1)
models.append('LDA', model2)
models.append('KNN', model3)
models.append('CART', model4)
models.append('NB', model5)
models.append('SVM', model6)

#Model Fit

# Classification
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
    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)
    print(confusion_matrix(y_train, cv_predict))
    print(f"{name}: {cv_res.mean()}: {cv_res.std()}")

#select  the model with good score.

#Regression
#same proc as classification algorithms can be followed to select appropriate algorithm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score

clf_lr = LinearRegression()
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,score='neg_mean_squared_error',verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Linear Regression #######')
print('Score : %.4f' % clf_lr.score(test_x, test_y))
print(accuracies)
print(f"{accuracies.mean()}: {accuracies.std()}")

mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
rmse = mean_squared_error(test_y, y_pred)**0.5
r2 = r2_score(test_y, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#Only ditc is sufficient. no need for list
params = [{'n_estimators':[1,10,15], 'max_depth':[1,2,3,4]}]
vc = GridSearchCV(estimator=RandomForestRegressor(n_estimators=1),param_grid=params,
                  verbose=1,cv=10,n_jobs=-1)
res = vc.fit(train_x,train_y)
print(res.best_score_)
print(res.best_params_)

OR
#For faster solutions . Randomly picks the values from params. Not an exhaustive one.
#But may not pick best solution

from sklearn.model_selection import RandomizedSearchCV
params = {'n_estimators':[1,10,15], 'max_depth':[1,2,3,4]}
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=n_iter_search, cv=5, iid=False)

print(random_search.best_score_)
print(random_search.best_params_)
     

clf_lr = RandomForestRegressor(n_estimators=10)
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,score='neg_mean_squared_error',verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Forest #######')
print('Score : %.4f' % clf_lr.score(test_x, test_y))
print(accuracies)
print(f"{accuracies.mean()}: {accuracies.std()}")

mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
rmse = mean_squared_error(test_y, y_pred)**0.5
r2 = r2_score(test_y, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

##Adjusted R2##

n=test_x.shape[0]
p=test_x.shape[1] - 1

adj_rsquared = 1 - (1 - r2) * ((n - 1)/(n-p-1))
adj_rsquared




######### saing the model ########333
from pickle import load
from pickle import dump

filename = 'mod_save.sav'
dump(clf_lr, open(filename,'wb'))


saved_model = load(open(filename,'rb'))
clf_lr.fit(test_x , test_y)

#######  OR #########
from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load 

# save the model to disk 
filename = 'finalized_model.sav' 
dump(model, filename)
# some time later...
# load the model from disk
loaded_model = load(filename) 
result = loaded_model.score(X_test, Y_test)
 print(result)