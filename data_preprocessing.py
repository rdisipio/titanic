import numpy as np
import pandas as pd

def process_embarkment(df):
   df['Embarked'].fillna('S', inplace=True )

   ports = {"S": 0, "C": 1, "Q": 2}
   df['Embarked'] = df['Embarked'].map(ports)
   print("INFO: port of embarkation (0=Southampton, 1=Cherbourg, 2=Queenstown):")
   print( df['Embarked'].value_counts() )

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

def process_cabin(df):
#   import re
   deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
   df['Cabin'].fillna("U0", inplace=True)
   df['Deck'] = df['Cabin'].map(lambda x: x[0])
#   df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
   df['Deck'] = df['Deck'].map(deck)
   df['Deck'] = df['Deck'].fillna(0)
   df['Deck'] = df['Deck'].astype(int)

   print("INFO: deck info" )
   print( df['Deck'].value_counts() )


def process_title(df):

   # 0=officer, 1=Royalty, 2=commoner
   title_dict = {
     "Capt": 0,
     "Col": 0,
     "Major": 0,
     "Dr": 0,
     "Rev": 0,
     "Jonkheer": 1,
     "Don": 1,
     "Sir" : 1,
     "the Countess": 1,
     "Lady" : 1,
     "Mme": 2,
     "Mlle": 2,
     "Ms": 2,
     "Mr" : 2,
     "Mrs" : 2,
     "Miss" : 2,
     "Master" : 2,
   }

   df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

   df['Title'] = df.Title.map(title_dict)

   print("INFO: title (0=officer, 1=royalty, 2=commoner):")
   print( df['Title'].value_counts() )
