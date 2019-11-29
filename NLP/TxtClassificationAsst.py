import pandas as pd
import numpy as np


df = pd.read_csv('D:\ML\ML_Rev\Datasets\moviereviews2.tsv',  sep='\t')

df.isnull().sum()

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pp = Pipeline([('tfid', TfidfVectorizer()),('svc', LinearSVC())])

pp.fit(X_train, y_train)

pred = pp.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

