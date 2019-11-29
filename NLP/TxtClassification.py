import pandas as pd
import numpy as np


df = pd.read_csv('D:\ML\ML_Rev\Datasets\smsspamcollection.tsv',  sep='\t')

df.isnull().sum()

from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

###########################  1  #####################
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_c = cv.fit_transform(X_train)
x_c.shape  #(4457, 7702)

from sklearn.feature_extraction.text import TfidfTransformer    
cv = TfidfTransformer()
x_c1 = cv.fit_transform(x_c)
x_c1.shape  #(4457, 7702)

###########################   2   #####################
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
x_t = tv.fit_transform(X_train)
x_t.shape

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(x_t, y_train)


###########################   3  #####################
from sklearn.pipeline import Pipeline
pp = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC())])

pp.fit(X_train, y_train)


pred = pp.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

pred1 = pp.predict([""])
print(pred1)
