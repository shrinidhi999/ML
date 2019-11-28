import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\ML\weather_data.csv.txt')
df['day'] = pd.to_datetime(df['day'])
df.info()

#filling missing cols, col by col
dfd = df.fillna({
        'temperature' : df['temperature'].median(),
        'windspeed' : df['windspeed'].median(),
        'event':0
        })

#filling missing numerical values with mean
n_df = df.select_dtypes('float64')
n_df = n_df.fillna(n_df.mean())
df.update(n_df)

#Filling missing cat values with high freq value
c_df = df.select_dtypes('object')
t = c_df['event'].value_counts()
t.index
t.values
k = t[t.values==t.max()].index[0]

c_df.fillna(t.index[t==t.max()][0],inplace=True)


for col in c_df.columns:
    t = c_df[col].value_counts()
    #tt=t.index[t==t.max()][0]
    c_df[col].fillna(t[t==t.max()].index[0],inplace=True)
c_df.update(c_df)

    
#Interpolate
i_df = df.interpolate()

t_df = df.set_index('day')
t_df = t_df.interpolate(methd='time')

c_df = t_df.select_dtypes('object')

#drop values
d_df = df.dropna(how='all')
d_df = df.dropna(thresh=3)

#replace values m
r_df= df.replace(np.nan,999)
r_df2= df.replace(np.NaN,{'temperature':df['temperature'].mean(), 'windspeed':888, 'event':'noo'})
df['event']

r_df3= df.replace(['Sunny','Snow','Rain','Cloudy'],[1,2,3,4])


df_dict={"temp":[1,2,3] , "day":[4,5,6]}
df_test = pd.DataFrame(df_dict)

df_test.extend({"temp":[22,44] , "day":[56,22]},ignore_index=True )

df['event2'] = "-".join(df['event'].str)


#drop
df1=df.drop(df.columns[1], axis=1)
df1


#Group by
df = dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\\titanic\\train.csv')
df.groupby('Pclass')['Survived'].sum()
df.groupby('Sex')['Survived'].mean()
df.groupby('Sex')['Survived'].sum()
df.groupby('SibSp')['Survived'].mean()

pd.date_range()


#lambda

df = pd.read_csv(r'D:\ML\weather_data.csv.txt')


df['event'] = df['event'] .fillna('Nope')
df['Upper'] = df['event'].apply(lambda x: x.upper())


df[df.columns[df.notnull().all()]]
df[df.columns[df.isnull().any()]]

#mean,mode, median, std
n1 = np.array([15, 15, 15, 14, 16])
print(n1.mean())
print(n1.std())
print(n1.var())

n2 = np.array([2, 7, 14, 22, 30.])
print(n2.mean())
print(n2.std())
print(n2.var())

a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])

from sklearn import preprocessing
import scipy.stats as stats
import matplotlib.pyplot as plt
#mm = preprocessing.MinMaxScaler(feature_range=(0,1))
#a1 = mm.fit_transform(a1)
#a1

mm = preprocessing.StandardScaler()
a1 = mm.fit_transform(a1)
a1

fit = stats.norm.pdf(a1, np.mean(a1), np.std(a1))  #this is a fitting indeed
plt.plot(a1,fit,'-o')

x = np.random.normal(size = 1000)
plt.hist(a1, normed=True, bins=30)


rl = zip([1,2,3],[4,5,6])
dict(rl)


###### Binning #############

df =pd.DataFrame({'age': np.random.randint(20,90,10)})
df['bins'] = pd.cut(x = df['age'], bins = [20,30,40,90], right=True)
df['bLabel'] = pd.cut(x = df['age'], bins = [20,30,40,90], right=True, labels=['20s','30s','above'])

##########   MAP #################3


df = pd.read_csv(r'D:\ML\weather_data.csv.txt')
event_num={'Rain':1,
           'Sunny':2,
           'Snow':3,
           'Cloudy':4,
           np.NaN:0
        }

df['eve_num'] = df['event'].map(event_num)

df['eve_num2'] = df['event'].replace(['Rain','Sunny','Snow','Cloudy', np.NaN],[1,2,3,4,0])




### Label Encoder

df = dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\\titanic\\train.csv')
df2 = df.select_dtypes(['object','category'])
df2.drop(['Ticket','Name','Cabin'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder


for i in range(len(df2.columns)):
    lb = LabelEncoder()
    df[df2.columns[i]] = lb.fit_transform(df[df2.columns[i]].astype(str))
    
    
######## scaling ###########    
from sklearn import preprocessing  
import matplotlib.pyplot as plt
  
x = np.array([[110],[123],[456]])
st = preprocessing.StandardScaler()
x_sc = st.fit_transform(x)

x_original = st.inverse_transform(x_sc)
dataset = pd.DataFrame({"Age":np.random.randint(20,90,10)})
# histograms 
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) 

dataset['Age1'] = st.fit_transform(np.array(dataset['Age']).reshape(-1,1))
dataset['Age2'] = preprocessing.MinMaxScaler().fit_transform(np.array(dataset['Age']).reshape(-1,1))

dataset['Age'].hist(xlabelsize=1, ylabelsize=1) 
dataset['Age1'].hist(xlabelsize=1, ylabelsize=1) 
dataset['Age2'].hist(xlabelsize=1, ylabelsize=1) 

dataset['Age'].plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) 
plt.show()
dataset['Age1'] = np.log(dataset['Age'])
dataset['Age1'].plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) 
plt.show()
dataset['Age2'] = np.cbrt(dataset['Age'])
dataset['Age2'].plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) 
plt.show()



df = dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\\titanic\\train.csv')
df.columns
df2 = df.loc[:,df.columns!='Survived']


###############  MAP ##############
df = pd.read_csv(r'D:\ML\weather_data.csv.txt')
df.head()

eve_dict = {"Rain":1,"Sunny":2,"Snow":3,"Cloudy":4,np.nan:5}

df['event'].replace(["Rain","Sunny","Snow","Cloudy",np.nan], [1,2,3,4,5], inplace=True)
df['New'] = df['event'].map(eve_dict)

#########################
df1 = pd.read_csv(r'D:\ML\weather_data.csv.txt')
df2 = pd.DataFrame()

df2['event'] = ["Rain","Sunny","Snow","Cloudy",np.nan]
df2['eve_num'] = [1,2,3,4,5]

df1['new'] = df1['event'].map(dict(zip(df2['event'],df2['eve_num'])))

################################################
df1=pd.DataFrame()
df1['col1'] = [1,2,3,4,5]
df1['col2'] = [1,3,7,5,9]

df1['col3'] = df1['col1'].isin(df1['col2'])

################################################
df = pd.read_csv(r'D:\ML\weather_data.csv.txt')

cnt = df['event'].value_counts()
cnt_indx = cnt[cnt.lt(2)].index

df.loc[df['event'].isin(cnt_indx), 'event'] = 'others'

df['event'].value_counts()[df['event']].values < 2




df = pd.DataFrame({'col1':[1,2,3] , 'col2':[10, 40, 90]})
np.mean(df['col2'])
np.std(df['col2'])

from scipy import stats
numeric_cols = df.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))


