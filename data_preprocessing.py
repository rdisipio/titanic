import numpy as np
import pandas as pd

def process_family(df):
   df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

   df['FamilySizeBin'] = df['FamilySize']
   df.loc[ df['FamilySizeBin'] == 1, 'FamilySizeBin' ] = 0
   df.loc[ (df['FamilySizeBin']>=2) & (df['FamilySizeBin']<=4), 'FamilySizeBin' ] = 1
   df.loc[ df['FamilySizeBin']>4, 'FamilySizeBin' ] = 2

   print("INFO: family size (0=single, 1=small family, 2=large family)" )
   print( df['FamilySizeBin'].value_counts() )

def process_age(df):
   print("INFO: fixing missing age info" )
   mean = df["Age"].mean()
   std = df["Age"].std()
   is_null = df["Age"].isnull().sum()
   # compute random numbers between the mean, std and is_null
   rand_age = np.random.randint(mean - std, mean + std, size = is_null)
   # fill NaN values in Age column with random values generated
   age_slice = df["Age"].copy()
   age_slice[np.isnan(age_slice)] = rand_age
   df["Age"] = age_slice
   df["Age"] = df["Age"].astype(int)

   df['AgeBin'] = df['Age'].astype(int)
   df.loc[ df['Age'] <= 11, 'AgeBin'] = 0
   df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'AgeBin'] = 1
   df.loc[(df['Age'] > 18) & (df['Age'] <= 25), 'AgeBin'] = 2
   df.loc[(df['Age'] > 25) & (df['Age'] <= 35), 'AgeBin'] = 3
   df.loc[(df['Age'] > 35) & (df['Age'] <= 45), 'AgeBin'] = 4
   df.loc[(df['Age'] > 45) & (df['Age'] <= 60), 'AgeBin'] = 5
   df.loc[ df['Age'] > 60, 'AgeBin'] = 6

   print("INFO: age category")
   print( df['AgeBin'].value_counts() )
