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

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale =1, color_codes=True) 


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

day_df = pd.read_csv(r'D:\ML\ML_Rev\Datasets\day.csv')# -*- coding: utf-8 -*-
hour_df = pd.read_csv(r'D:\ML\ML_Rev\Datasets\hour.csv')# -*- coding: utf-8 -*-

day_df.drop(['dteday','instant','weekday','workingday'], axis=1, inplace=True)
hour_df.drop(['dteday','instant','weekday','workingday'], axis=1, inplace=True)

day_df.isnull().sum()
hour_df.isnull().sum()

print(sum(day_df.duplicated(day_df.columns)))
print(sum(hour_df.duplicated(hour_df.columns)))

hour_df = hour_df.drop_duplicates(hour_df.columns, keep='last')



