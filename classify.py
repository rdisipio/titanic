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
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from data_preprocessing import *

#import tensorflow as tf

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
all_df = merge_datasets(train_df, test_df)

# print out example
print("INFO: training sample size:", len(train_df))
print("INFO: testing sample size:", len(test_df))

process_title(all_df)
for data in [train_df, test_df]:
    # order matters!

    process_family(data)
    process_cabin(data)
    process_embarkment(data)
    process_title(data)
    process_fare(data, all_df)
    process_gender(data)
    process_age(data, all_df)

print(train_df.describe())

features = ['Pclass', 'Sex', 'Embarked',
            'FamilySize',  # 'Parch', 'SibSp',
            #'Singleton', 'LargeFamily', 'SmallFamily',
            'Deck', 'Title',
            'Fare',  # 'FareBin',
            'Age',  # 'AgeBin',
            ]
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
cv_rf = cross_val_score(rf, X_train, Y_train, cv=20, scoring="accuracy")
mean_rf = round(cv_rf.mean(), 2)
std_rf = round(cv_rf.std(), 2)

# Logistic regression
logreg = LogisticRegression(solver='liblinear')  # 'lbfgs')
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
cv_log = cross_val_score(logreg, X_train, Y_train, cv=20, scoring="accuracy")
mean_log = round(cv_log.mean(), 2)
std_log = round(cv_log.std(), 2)

# Naive Bayes
bayes = GaussianNB()
bayes.fit(X_train, Y_train)
Y_pred_bayes = bayes.predict(X_test)
acc_bayes = round(bayes.score(X_train, Y_train) * 100, 2)
cv_bayes = cross_val_score(bayes, X_train, Y_train, cv=20, scoring="accuracy")
mean_bayes = round(cv_bayes.mean(), 2)
std_bayes = round(cv_bayes.std(), 2)

# MLP
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
cv_mlp = cross_val_score(mlp, X_train, Y_train, cv=20, scoring="accuracy")
mean_mlp = round(cv_mlp.mean(), 2)
std_mlp = round(cv_mlp.std(), 2)

# XGBoost
#objective = "binary:hinge"
objective = "binary:logistic"
bdt = XGBClassifier(objective=objective, max_depth=6)
bdt.fit(X_train, Y_train)
Y_pred_bdt = bdt.predict(X_test)
acc_bdt = round(accuracy_score(bdt.predict(X_train), Y_train) * 100, 2)
cv_bdt = cross_val_score(bdt, X_train, Y_train, cv=20, scoring="accuracy")
mean_bdt = round(cv_bdt.mean(), 2)
std_bdt = round(cv_bdt.std(), 2)

# Put results together
results = pd.DataFrame({
    'Model': ['RandomForest', 'LogisticRegression', 'NaiveBayes', 'MLP', 'BDT'],
    'Acc': [acc_rf, acc_log, acc_bayes, acc_mlp, acc_bdt],
    'CVmean': [mean_rf, mean_log, mean_bayes, mean_mlp, mean_bdt],
    'CVstd': [std_rf, std_log, std_bayes, std_mlp, std_bdt]
})

result_df = results.sort_values(by='CVmean', ascending=False)
result_df = result_df.set_index('CVmean')
print(result_df.head(9))


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
