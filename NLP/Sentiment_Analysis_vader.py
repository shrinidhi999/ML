
import nltk
nltk.download('vader_lexicon')


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


a = 'this is a good movie'
print(sid.polarity_scores(a))

a = 'this is a GOOD movie'
print(sid.polarity_scores(a))

a = 'this is the best movie'
print(sid.polarity_scores(a))

a = 'this is a good movie!!!!!'
print(sid.polarity_scores(a))

a = 'this is a bad movie'
sid.polarity_scores(a)

#################################################################################

import pandas as pd
import numpy as np


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

df = pd.read_csv(r'D:\ML\ML_Rev\Datasets\amazonreviews.tsv',  sep='\t')

df['label'].value_counts()

df.isnull().sum()

rev_list=[]
for ind, label, rev in df.itertuples():
    if type(rev) == str and rev.isspace():
           rev_list.append(ind)     
    

df.drop(rev_list, inplace=True)


df['Score'] = df['review'].apply(lambda x: sid.polarity_scores(x))
df['Score'] = df['Score'].apply(lambda x: x['compound'])
df['Pred'] = df['Score'].apply(lambda x: 'pos' if x>0 else 'neg')


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(df['label'], df['Pred']))
print(confusion_matrix(df['label'], df['Pred']))
print(classification_report(df['label'], df['Pred']))