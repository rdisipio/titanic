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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from data_preprocessing import *

#import tensorflow as tf

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# print out example
print("INFO: training sample size:", len(train_df))
print("INFO: testing sample size:", len(test_df))

for data in [train_df, test_df]:
    # order matters!

    process_family(data)
    process_cabin(data)
    process_embarkment(data)
    process_title(data)
    process_fare(data)
    process_gender(data)
    process_age(data)

print(train_df.describe())

features = ['Pclass', 'Sex', 'Embarked',
            'FamilySize', 'Parch', 'SibSp',
            'Deck', 'Title',
            'Fare', 'FareBin',
            'Age', 'AgeBin', ]
X_train = train_df[features]
Y_train = train_df["Survived"]
X_test = test_df[features]

# Apply standardization?
#scaler = StandardScaler()
#scaler = MinMaxScaler([-1, 1])
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#X_train = pd.DataFrame(X_train)
#X_test = pd.DataFrame(X_test)

# Apply PCA?
#pca = PCA(n_components=8)
#pc_train = pca.fit_transform(X_train)
#X_train = pd.DataFrame(data = pc_train)
#pc_test = pca.transform(X_test)
#X_test  = pd.DataFrame(data = pc_test)

# Check invalid fields
# print(features)
# for f in features:
#   print( "train", f, X_train[f].isnull().sum(), np.isnan(X_train[f]).sum() )
#   print( "test", f, X_test[f].isnull().sum(), np.isnan(X_test[f]).sum() )

# Random Forest
params = {'bootstrap': True,
          'max_depth': 6,
          'max_features': 'log2',
          'min_samples_leaf': 3,
          'min_samples_split': 2,
          'n_estimators': 100}
rf = RandomForestClassifier(**params)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
rf.score(X_train, Y_train)
acc_rf = round(rf.score(X_train, Y_train) * 100, 2)

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Naive Bayes
bayes = GaussianNB()
bayes.fit(X_train, Y_train)
Y_pred_bayes = bayes.predict(X_test)
acc_bayes = round(bayes.score(X_train, Y_train) * 100, 2)

# XGBoost
#objective = "binary:hinge"
objective = "binary:logistic"
bdt = XGBClassifier(objective=objective, max_depth=6)
bdt.fit(X_train, Y_train)
Y_pred_bdt = bdt.predict(X_test)
acc_bdt = round(accuracy_score(bdt.predict(X_train), Y_train) * 100, 2)

# Put results together
results = pd.DataFrame({
    'Model': ['RandomForest', 'LogisticRegression', 'NaiveBayes', 'BDT'],
    'Score': [acc_rf, acc_log, acc_bayes, acc_bdt]
})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))

scores = cross_val_score(bdt, X_train, Y_train, cv=10, scoring="accuracy")
print("INFO: cross-validation")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

print("INFO: features ranking: Random Forest")
importances = pd.DataFrame(
    {'feature': X_train.columns, 'importance': np.round(rf.feature_importances_, 3)})
importances = importances.sort_values(
    'importance', ascending=False).set_index('feature')
print(importances.head(len(features)))

print("INFO: features ranking: Boosted Trees")
importances = pd.DataFrame(
    {'feature': X_train.columns, 'importance': np.round(bdt.feature_importances_, 3)})
importances = importances.sort_values(
    'importance', ascending=False).set_index('feature')
print(importances.head(len(features)))

print("INFO: preparing submission file:")
fname = "data/submission.csv"
#df = pd.DataFrame([test_df["PassengerId"], Y_pred_rf])
df = pd.DataFrame([test_df["PassengerId"], Y_pred_bdt])
df = df.transpose()
df.columns = ["PassengerId", "Survived"]
df.set_index("PassengerId", inplace=True)
print(df.head(10))
df.to_csv(fname, header=True, index=True)
