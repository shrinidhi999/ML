import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

#%reset -f
dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\KM_Train.csv')

#Pandas profile
#import pandas_profiling
#profile = pandas_profiling.ProfileReport(dataset)
#print(profile)

#Taking care of Missing Data in Dataset
print(dataset.isnull().sum())


#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

dataset.info

df = dataset.select_dtypes(include=['category','object'])
Labels = dataset['activity']
dataset.drop(['activity','rn'],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(dataset)

from sklearn.decomposition import PCA
pca= PCA(n_components=None)
dataset = pca.fit_transform(dataset)
e_var = pca.explained_variance_ratio_
e_var =e_var.reshape(-1,1)


pca= PCA(n_components=1)
dataset = pca.fit_transform(dataset)
e_var1 = pca.explained_variance_ratio_
e_var1 =e_var1.reshape(-1,1)

wcss=[]
from sklearn.cluster import KMeans
for i in range(1,9):
    km = KMeans(n_clusters=i)
    km.fit_predict(dataset)
    wcss.append(km.inertia_)
    
#
#plt.plot(range(1,9) , wcss)    
#plt.show()

from IPython.display import display
plt.figure(figsize=(12,8)) 
km = KMeans(n_clusters=2, random_state=42, n_init=30)
pred = km.fit_predict(dataset)
c_labels = km.labels_
df = pd.DataFrame({'clust_label': c_labels, 'orig_label': Labels.tolist()})
ct = pd.crosstab(df['clust_label'], df['orig_label'])
#    y_clust = k_means.predict(data_frame)
display(ct)

from sklearn.metrics import silhouette_score
print(f"silhouette : {silhouette_score(dataset, pred, metric='euclidean')}")





########################## Hierarchical ############################################
#%reset -f
dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\KM_Train.csv')

#Taking care of Missing Data in Dataset
print(dataset.isnull().sum())

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

dataset.info

df = dataset.select_dtypes(include=['category','object'])
Labels = dataset['activity']
dataset.drop(['activity','rn'],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(dataset)

from sklearn.decomposition import PCA
pca= PCA(n_components=1)
dataset = pca.fit_transform(dataset)
e_var1 = pca.explained_variance_ratio_
e_var1 =e_var1.reshape(-1,1)

import scipy.cluster.hierarchy as sh
d = sh.dendrogram(sh.linkage(dataset, method='ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering, Di 
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
pred = hc.fit_predict(dataset)

from sklearn.metrics import silhouette_score
print(f"silhouette : {silhouette_score(dataset, pred, metric='euclidean')}")
