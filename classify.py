#!/usr/bin/env python3

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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
   process_gender(data)

print(train_df.describe() )

features = [ 'Pclass', 'Sex', 'Embarked', 'FamilySizeBin', 'AgeBin', 'Deck', 'Title', 'FareBin' ]
to_drop = [ "Name", "PassengerId", "Fare", "Age", "Cabin", "Ticket", 'SibSp', 'Parch']
X_train = train_df[features]
Y_train = train_df["Survived"]
X_test  = test_df[features]

# Check invalid fields
#print(features)
#for f in features:
#   print( "train", f, X_train[f].isnull().sum(), np.isnan(X_train[f]).sum() )
#   print( "test", f, X_test[f].isnull().sum(), np.isnan(X_test[f]).sum() )

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Put results together
results = pd.DataFrame({
    'Model': [ 'RandomForest', 'LogisticRegression', 'NaiveBayes' ],
    'Score': [ acc_random_forest, acc_log, acc_gaussian ]
    })

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))
