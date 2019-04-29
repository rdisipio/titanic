import numpy as np
import pandas as pd


def merge_datasets(df_train, df_test):
    df_all = df_train.copy()
    targets = df_train.Survived
    df_all.drop(['Survived'], 1, inplace=True)
    df_all.append(df_test)
    df_all.reset_index(inplace=True)
    df_all.drop(['index', 'PassengerId'], inplace=True, axis=1)
    return df_all


def process_gender(df):
    gender_dict = {"male": 0, "female": 1}
    df['Sex'] = df['Sex'].map(gender_dict)


def process_embarkment(df):
    df['Embarked'].fillna('S', inplace=True)

    ports = {"S": 0, "C": 1, "Q": 2}
    df['Embarked'] = df['Embarked'].map(ports)


def process_family(df):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    df['FamilySizeBin'] = df['FamilySize']
    df.loc[df['FamilySizeBin'] == 1, 'FamilySizeBin'] = 0
    df.loc[(df['FamilySizeBin'] >= 2) & (
        df['FamilySizeBin'] <= 4), 'FamilySizeBin'] = 1
    df.loc[df['FamilySizeBin'] > 4, 'FamilySizeBin'] = 2

    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


def process_age(df, df_all):

    # age has a non-uniform distribution, depends on sex, ticket class and title
    df['Age'] = df_all.groupby(['Sex', 'Pclass', 'Title'])[
        'Age'].transform(lambda x: x.fillna(x.median()))

    df['AgeBin'] = df['Age'].astype(int)
    df.loc[df['Age'] <= 11, 'AgeBin'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'AgeBin'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 25), 'AgeBin'] = 2
    df.loc[(df['Age'] > 25) & (df['Age'] <= 35), 'AgeBin'] = 3
    df.loc[(df['Age'] > 35) & (df['Age'] <= 45), 'AgeBin'] = 4
    df.loc[(df['Age'] > 45) & (df['Age'] <= 60), 'AgeBin'] = 5
    df.loc[df['Age'] > 60, 'AgeBin'] = 6


def process_fare(df, df_all):
    # fare has a non-uniform distribution, depends on sex, ticket class and title
    df_all.Fare.fillna(df_all.Fare.mean(), inplace=True)  # fix missing entries
    df['Fare'] = df_all.groupby(['Sex', 'Pclass', 'Title'])[
        'Fare'].transform(lambda x: x.fillna(x.median()))

    df.loc[df['Fare'] <= 10, 'FareBin'] = 0
    df.loc[(df['Fare'] > 10) & (df['Fare'] <= 20), 'FareBin'] = 1
    df.loc[(df['Fare'] > 20) & (df['Fare'] <= 30), 'FareBin'] = 2
    df.loc[(df['Fare'] > 30) & (df['Fare'] <= 50), 'FareBin'] = 3
    df.loc[(df['Fare'] > 50) & (df['Fare'] <= 100), 'FareBin'] = 4
    df.loc[(df['Fare'] > 100) & (df['Fare'] <= 200), 'FareBin'] = 5
    df.loc[df['Fare'] > 200, 'FareBin'] = 6
    df['FareBin'] = df['FareBin'].astype(int)


def process_cabin(df):
        #   import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df['Cabin'].fillna("U0", inplace=True)
    df['Deck'] = df['Cabin'].map(lambda x: x[0])
#   df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(deck)
    df['Deck'] = df['Deck'].fillna(0)
    df['Deck'] = df['Deck'].astype(int)


def process_title(df):

   # 0=officer, 1=Royalty, 2=commoner(male), 3=commoner(female)
    title_dict = {
        "Officer": 0,
        "Royalty": 1,
        "Mr": 2,
        "Miss": 3,
        "Mrs": 4,
        "Master": 5,
    }

    # 18,1,2,"Williams, Mr. Charles Eugene",male,,0,0,244373,13,,S

    df['Title'] = df['Name'].map(lambda name: name.split(',')[
                                 1].split('.')[0].strip())

    df.Title.replace(to_replace=['Dr', 'Rev', 'Col',
                                 'Major', 'Capt'], value='Officer', inplace=True)
    df.Title.replace(to_replace=['Dona', 'Jonkheer', 'Countess',
                                 'Sir', 'Lady', 'Don'], value='Royalty', inplace=True)
    df.Title.replace({'Mlle': 'Miss', 'Ms': 'Miss',
                      'Mme': 'Mrs'}, inplace=True)

    df['Title'] = df.Title.map(title_dict)
    # default to most probable (male)
    df['Title'] = df['Title'].fillna(title_dict['Mr'])
