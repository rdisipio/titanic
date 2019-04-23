#!/usr/bin/env python3

import numpy as np
import pandas as pd

#import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

# print out example
print("INFO: training sample size:", len(train_df) )
print(train_df.head(5))

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print("INFO: missing data:")
print(missing_data.head(5))
