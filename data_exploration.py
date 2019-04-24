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

params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [10, 6]
}
plt.rcParams.update(params)

data = pd.read_csv("data/train.csv")

# print out example
print("INFO: training sample size:", len(data) )
print(data.head(5))

print("INFO: mean survival rate:", data['Survived'].mean() )

# add custom fields
data['Died'] = 1 - data['Survived']

process_family( data )
process_age(data)
process_cabin( data )
process_embarkment( data )
process_title(data)

#plt.figure()

#plot_1 = data.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True)
#plt.savefig("fig_01.png")

#plot_2 = sns.violinplot(x='Sex', y='Age', hue='Survived',data=data,split=True)
#plt.savefig("fig_02.png")

#xedges = (0,10,20,30,40,50,100,500)
#plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
#         stacked=True,
#         bins = 20, label = ['Survived','Dead'])
#plt.xlabel('Fare')
#plt.ylabel('Number of Passengers')
#plt.legend()
#plt.savefig("fig_03.png")
