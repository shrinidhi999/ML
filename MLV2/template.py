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

#%reset -f


def Extract_Dataset():
    dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\cereal.csv')
    return dataset

def set_display():
    # Display propertice
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    dataset.describe()

def Pandas_Profile():
    #Pandas profile
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(dataset)
    print(profile)

def Missing_Values():
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
    return col

def Duplicate_Check():
    #Duplicate check
    print(sum(dataset.duplicated(dataset.columns)))
    dataset = dataset.drop_duplicates(dataset.columns, keep='last')
    return dataset

def Plots():
    # box and whisker plots 
    numeric_cols = dataset.select_dtypes(include=['int64','float64'])
    numeric_cols.plot(kind='box', subplots=True, layout=(13,13), sharex=False, sharey=False) 
    plt.show()

    # histograms 
    numeric_cols.hist() 
    plt.show()

def skew_kurtosis():
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

def Outlier_Remove():
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

def Outlier_Remove_2():
#    LINK: https://towardsdatascience.com/unsupervised-learning-project-creating-customer-segments-17c4b4bbf925
    outliers = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
   # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3-Q1)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    lista = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist()
    outliers.append(lista)
    
    
    # Detecting outliers that appear in more than one product
    seen = {}
    dupes = []
    
    for lista in outliers:
        for index in lista:
            if index not in seen:
                seen[index] = 1
            else:
                if seen[index] == 1:
                    dupes.append(index)
                seen[index] += 1
    dupes = sorted(dupes)
    dupes
    
    # Removing outliers  
    good_data = log_data.drop(dupes, axis=0).reset_index(drop=True)
    
def outlier_method3_winsorize():    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    from scipy.stats import mstats
    X_train[num_cols].skew()
    
    for feature in X_train[num_cols].keys():
        sk = X_train[feature].skew()
        if sk > 2 or sk < -2:
            print(feature)
            X_train[feature] = pd.Series(mstats.winsorize(X_train[feature], limits=[0.05, 0.05]) )    
    
#    Datapoints considered outliers that are present in more than one feature
    
def Corr_Plot():
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

    col = corr.index
    for i in range(len(col)):
        for j in range(i+1, len(col)):
            if corr.iloc[i,j] >= 0.8:
                print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")

    dataset.drop(['potass','weight'],inplace=True,axis=1)

def train_test():

    X= dataset[:,1:2]
    y= dataset[:,-1]
    #Splitting the Dataset into Training set and Test Set
    #stratify parameter ll divide the dataset such thtat- y valyes are proprtionately divided among the test and train sets
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7, stratify=y)
    return X_train, X_test, y_train



def Category_freq_chk():        
    #For Categorical vars - remove/replace low freq vars

    dataset['native'] = dataset['native'].apply(lambda x: 'other' if cnt1[x] <= 100 else x)

    for col in dataset.select_dtypes(include=['category','object']).columns:
        dataset.loc[dataset[col].value_counts()[dataset[col]].values < 10, col] = np.nan

    dataset['mfr'].value_counts()[dataset['mfr']].values < 10


    ## To replace categorical values with less than 10 freq

    cnt = df['col'].value_counts()
    cnt_indx = cnt[cnt < 10].index
    df.loc[df['col'].isin(cnt_indx), 'col'] = 'others'  

def get_dummies():
    #Encoding categorical data
    #drop_first - to avoid dummy variable trap
    dataset = pd.get_dummies(dataset, drop_first=True)
    return dataset

def Variance_Check():
    # Variance check - if variance less than threshold remove them
    from sklearn.feature_selection import VarianceThreshold
    constant_filter = VarianceThreshold(threshold=0.5)
    constant_filter.fit(dataset)
    print(dataset.columns[constant_filter.get_support()])
    constant_columns = [column for column in dataset.columns if column not in dataset.columns[constant_filter.get_support()]]
    dataset.drop(labels=constant_columns, axis=1, inplace=True)

def Feature_Scaling():
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def PCA():
    #PCA
    from sklearn.decomposition import PCA
    pca= PCA(n_components=None)
    x = pca.fit_transform(x)
    e_var = pca.explained_variance_ratio_
    e_var =e_var.reshape(-1,1)

def RFECV():
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

def handle_datetime_cols():
    earliest = min(dataset['last_review'])
    dataset['last_review'].fillna(earliest, inplace=True)
    dataset['last_review'] = dataset['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())
    
#    x.toordinal() - earliest.toordinal() -- converts datetime to numerical vals
###########  REGRESSION ##########
def initial_models():        
    # Split-out validation dataset 
    array = dataset.values 
    X = array[:,0:13] 
    Y = array[:,13] 
    validation_size = 0.20 
    seed = 7 
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric 
    num_folds = 10 
    seed = 7 
    scoring = 'neg_mean_squared_error'

    # Spot-Check Algorithms 
    models = [] 
    models.append(('LR', LinearRegression())) 
    models.append(('LASSO', Lasso())) 
    models.append(('EN', ElasticNet())) 
    models.append(('KNN', KNeighborsRegressor())) 
    models.append(('CART', DecisionTreeRegressor())) 
    models.append(('SVR', SVR()))

    results = [] 
    names = [] 
    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)

def pipeline_models():
        ##################################################################33

    # Standardize the dataset 
    # create pipeline 
    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('lr', LinearRegression())) 
    model1 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('Lasso', Lasso())) 
    model2 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('ElasticNet', ElasticNet())) 
    model3 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('KNN', KNeighborsRegressor())) 
    model4 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('DecisionTreeRegressor', DecisionTreeRegressor())) 
    model5 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('SVR', SVR())) 
    model6 = Pipeline(estimators) 

    models =[]
    models.append(('LR', model1))
    models.append(('Lasso', model2))
    models.append(('ElasticNet', model3))
    models.append(('KNN', model4))
    models.append(('DecisionTreeRegressor', model5))
    models.append(('SVR', model6))

    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)

def Grid_search():
        
    # LR Algorithm tuning 
    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    rescaled_XV = scaler.transform(X_validation)
    model = LinearRegression() 
    model.fit(rescaledX, Y_train)
    pred = model.predict(rescaled_XV)

    print(mean_squared_error(Y_validation, pred) ** 0.5)
    #0.056205607054542965

    print('')
    print('####### Linear Regression #######')
    print('Score : %.4f' % model.score(rescaled_XV, Y_validation))

    rmse = mean_squared_error(Y_validation, pred)**0.5
    r2 = r2_score(Y_validation, pred)

    print('')
    print('RMSE   : %0.2f ' % rmse)
    print('R2     : %0.2f ' % r2)

def ensemble():
            
    ### ENSEMBLE #############33

    # create pipeline 
    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('ADA', AdaBoostRegressor())) 
    model1 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('GDB', GradientBoostingRegressor())) 
    model2 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('RF', RandomForestRegressor())) 
    model3 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('ET', ExtraTreesRegressor())) 
    model4 = Pipeline(estimators) 


    models =[]
    models.append(('ADA', model1))
    models.append(('GDB', model2))
    models.append(('RF', model3))
    models.append(('ET', model4))

    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)    

def ensemble_tuning():
            
    # GDB Algorithm tuning 
    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    param_grid = {"n_estimators":[50,100,150,200,250,300,350,400,500,600]}
    model = GradientBoostingRegressor(random_state=seed) 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
    grid_result = grid.fit(rescaledX, Y_train)    


    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
    means = grid_result.cv_results_['mean_test_score'] 
    stds = grid_result.cv_results_['std_test_score'] 
    params = grid_result.cv_results_['params'] 
    for mean, stdev, param in zip(means, stds, params): 
        print("%f (%f) with: %r" % (mean, stdev, param))

def Prediction_Accuracy():      

    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    model = GradientBoostingRegressor(n_estimators = 300, random_state=seed)         
    model.fit(rescaledX, Y_train)

    rescaledX_validation = scaler.transform(X_validation) 
    pred = model.predict(rescaledX_validation)

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

###########  Classification ##########
def initial_models_class():

    # Test options and evaluation metric 
    num_folds = 10 
    seed = 7 
    scoring = 'accuracy'


    # Spot-Check Algorithms 
    models = [] 
    models.append(('LR', LogisticRegression())) 
    models.append(('LDA', LinearDiscriminantAnalysis())) 
    models.append(('KNN', KNeighborsClassifier())) 
    models.append(('CART', DecisionTreeClassifier())) 
    models.append(('NB', GaussianNB())) 
    #models.append(('SVM', SVC()))

    results = [] 
    names = [] 
    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)

def pipeline_models_class():
        
    # Standardize the dataset 
    # create pipeline 
    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('lr', LogisticRegression())) 
    model1 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis())) 
    model2 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('KNeighborsClassifier', KNeighborsClassifier())) 
    model3 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('DecisionTreeClassifier', DecisionTreeClassifier())) 
    model4 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('GaussianNB', GaussianNB())) 
    model5 = Pipeline(estimators) 
    #
    #estimators = [] 
    #estimators.append(('standardize', StandardScaler())) 
    #estimators.append(('SVC', SVC())) 
    #model6 = Pipeline(estimators) 

    models =[]
    models.append(('LR', model1)) 
    models.append(('LDA', model2)) 
    models.append(('KNN', model3))
    models.append(('CART', model4)) 
    models.append(('NB', model5)) 
    #models.append(('SVM', model6))   

    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)
    
def Grid_search_class():
        
    # LR Algorithm tuning 
    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "penalty" : ['l1', 'l2'], "solver" : ['liblinear', 'warn']}
    model = LogisticRegression() 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
    grid_result = grid.fit(rescaledX, y_train)    


    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

    rescaled_X_test = scaler.transform(X_test) 
    model = LogisticRegression(C= 0.3,penalty= 'l2',solver='liblinear')
    model.fit(rescaledX, y_train)

    pred = model.predict(rescaled_X_test)

    print(f"Accurancy: {accuracy_score(y_test,pred)} ")
    print(f"Accurancy: {confusion_matrix(pred,y_test)} ")
    print(f"Accurancy: {classification_report(pred,y_test)} ")

    #Accurancy: 0.9086861491135757

def ensemble_class():
       
    ### ENSEMBLE #############33

    # create pipeline 
    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('ADA', AdaBoostRegressor())) 
    model1 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('GDB', GradientBoostingRegressor())) 
    model2 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('RF', RandomForestRegressor())) 
    model3 = Pipeline(estimators) 

    estimators = [] 
    estimators.append(('standardize', StandardScaler())) 
    estimators.append(('ET', ExtraTreesRegressor())) 
    model4 = Pipeline(estimators) 


    models =[]
    models.append(('ADA', model1))
    models.append(('GDB', model2))
    models.append(('RF', model3))
    models.append(('ET', model4))

    for name, model in models: 
        kfold = KFold(n_splits=num_folds, random_state=seed) 
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
        results.append(cv_results) 
        names.append(name) 
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
        print(msg)    

def ensemble_tuning_class():
            
    # GDB Algorithm tuning 
    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    param_grid = {"n_estimators":[50,100,150,200,250,300,350,400,500,600]}
    model = GradientBoostingRegressor(random_state=seed) 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
    grid_result = grid.fit(rescaledX, Y_train)    


    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
    #Best: -0.004372 using {'n_estimators': 50}    

def Prediction_Accuracy_class():
    
    scaler = StandardScaler().fit(X_train) 
    rescaledX = scaler.transform(X_train) 
    model = GradientBoostingRegressor(n_estimators = 50, random_state=seed)         
    model.fit(rescaledX, Y_train)

    rescaledX_validation = scaler.transform(X_validation) 
    pred = model.predict(rescaledX_validation)
    print(mean_squared_error(Y_validation, pred) ** 0.5)

    #0.06071023441735939 

    print('')
    print('####### GBR #######')
    print('Score : %.4f' % model.score(rescaledX_validation, Y_validation))

    rmse = mean_squared_error(Y_validation, pred)**0.5
    r2 = r2_score(Y_validation, pred)

    print('')
    print('RMSE   : %0.2f ' % rmse)
    print('R2     : %0.2f ' % r2)

def Model_Save():      

    ######### saving the model ########333
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




#######    Method calls   #############
dataset = Extract_Dataset()
set_display()
Pandas_Profile()
col = Missing_Values()
dataset = Duplicate_Check()
Plots()
Corr_Plot()
Outlier_Remove()
Category_freq_chk()
dataset = get_dummies()
Variance_Check()
skew_kurtosis() 
X_train, X_test, y_train = train_test()
X_train, X_test = Feature_Scaling()
PCA()
RFECV()

###########  REGRESSION ##########

###########  Classification ##########

Model_Save()

