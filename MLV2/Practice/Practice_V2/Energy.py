import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from scipy import stats
from sklearn.preprocessing import Imputer

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale =1, color_codes=True) 


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\Energy.csv')# -*- coding: utf-8 -*-

dataset['ENERGY STAR Score'].value_counts()

dataset = dataset.replace({'Not Available': np.nan})

dataset_test = dataset[dataset['ENERGY STAR Score'].isnull()]

dataset = dataset[~dataset['ENERGY STAR Score'].isnull()]

dataset.dtypes


#Convert Data to Correct Data Types    
num_indicaters = ['ft²', 'kBtu', 'Metric Tons CO2e', 'kWh', 'therms', 'gal', 'Score']    
for col in dataset.columns:
    if any(col for s in num_indicaters if s in col):
        dataset[col] = dataset[col].astype('float64')     

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

chk = dataset.isnull().sum()*100/len(dataset)
chk1  = chk[chk.apply(lambda x: x >50)].index
dataset.drop(chk1, axis=1, inplace=True)

n_col = dataset.select_dtypes(include=['float64']).columns
c_col = dataset.select_dtypes(include=['object']).columns
imp = Imputer(strategy='median')
dataset[n_col] = imp.fit_transform(dataset[n_col])
        
c_dict={}
for i in c_col:
    c_dict[i] = dataset[i].mode()
    

for i in c_col:
    dataset[i].fillna(c_dict[i][0], inplace=True)
    

######Visualisations###

plt.hist(dataset['ENERGY STAR Score'].dropna())
plt.hist(dataset['Site EUI (kBtu/ft²)'].dropna())
        
dataset['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10)

dataset[dataset['Site EUI (kBtu/ft²)'] == 869265.0]

#####Remove outliers####
num_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
from scipy.stats import mstats
dataset[num_cols].skew()

for feature in num_cols:
    sk = dataset[feature].skew()
    if sk > 2 or sk < -2:
        print(feature)
        dataset[feature] = pd.Series(mstats.winsorize(dataset[feature], limits=[0.05, 0.05]) ) 
        

##Scaling
sc = StandardScaler()
dataset[num_cols] = sc.fit_transform(dataset[num_cols])

        
####Correlation
import phik
from phik import resources, report

corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


corr = corr['ENERGY STAR Score'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.1]
dataset_train.drop(to_drop_1, axis=1, inplace=True)

corr = dataset_train.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

col = corr.index
corr_target = corr['RainTomorrow'].abs()
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.7:
            if corr_target[col[i]] > corr_target[col[j]]:
                print(col[j])
                dataset_train.drop(col[j], axis=1, inplace=True)
            else:
                print(col[i])
                dataset_train.drop(col[i], axis=1, inplace=True)


