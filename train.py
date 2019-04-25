#!/usr/bin/env python3

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from data_preprocessing import *

#import tensorflow as tf

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# print out example
print("INFO: training sample size:", len(train_df) )
print("INFO: testing sample size:", len(test_df) )

for data in [ train_df, test_df ]:
   process_family( data )
   process_age(data)
   process_cabin( data )
   process_embarkment( data )
   process_title(data )
   process_fare(data)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


