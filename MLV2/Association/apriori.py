# Regression Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.\Datasets\Market_Basket_Optimisation.csv', header=None)
trans = []

for i in range(0, 7501):
    trans.append([str(dataset.values[i,j]) for j in range(0, 20)])

from MLV2.Association.apyori import apriori
rules = apriori(trans, min_support=0.003, min_confidence=.2,min_lift=3,min_length=2)

print(type(rules))
res = list(rules)


