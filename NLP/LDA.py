import pandas as pd
import numpy as np


npr = pd.read_csv(r'D:\ML\ML_Rev\Datasets\npr.csv')


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(npr['Article'])

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=7, random_state=42)

lda.fit(dtm)

len(cv.get_feature_names())

cv.get_feature_names()[1234]

len(lda.components_)    

lda.components_

single_topic = lda.components_[0]

single_topic.argsort()

top_word_indices = single_topic.argsort()[-10:]

for i in top_word_indices:
    print(cv.get_feature_names()[i])


for ind, topic in enumerate(lda.components_):
    print(f"Topic : {ind}")
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print("\n\n")


dtm.shape

topic_results = lda.transform(dtm)

len(topic_results)

topic_results[6].argmax()

npr['topic'] = topic_results.argmax(axis=1)
