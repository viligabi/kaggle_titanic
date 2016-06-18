#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import tree
from sklearn import grid_search
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)
    else:
        return "Number of predictions does not match number of outcomes!"

def optimize_data(df_):
    df = df_.copy()
    try:
        df.drop(['Ticket'], inplace=True, axis=1)
    except: pass
    try:
        df.drop(['Name'], inplace=True, axis=1)
    except:
        pass
    try:
        df.drop(["PassengerId"], inplace=True, axis=1)
    except:
        pass
    try:
        df.drop(["Survived"], inplace=True, axis=1)
    except:
        pass
    df.loc[df.Sex != 'male', 'Sex'] = 0
    df.loc[df.Sex == 'male', 'Sex'] = 1
    df.Age = df.Age.fillna(df.Age.mean())
    df.Cabin.fillna('0', inplace=True)
    df.loc[df.Cabin.str[0] == 'A', 'Cabin'] = 1
    df.loc[df.Cabin.str[0] == 'B', 'Cabin'] = 2
    df.loc[df.Cabin.str[0] == 'C', 'Cabin'] = 3
    df.loc[df.Cabin.str[0] == 'D', 'Cabin'] = 4
    df.loc[df.Cabin.str[0] == 'E', 'Cabin'] = 5
    df.loc[df.Cabin.str[0] == 'F', 'Cabin'] = 6
    df.loc[df.Cabin.str[0] == 'G', 'Cabin'] = 7
    df.loc[df.Cabin.str[0] == 'T', 'Cabin'] = 8
    df.Embarked.fillna(0, inplace=True)
    df.loc[df.Embarked == 'C', 'Embarked'] = 1
    df.loc[df.Embarked == 'Q', 'Embarked'] = 2
    df.loc[df.Embarked == 'S', 'Embarked'] = 3
    df.fillna(-1, inplace=True)
    return df.astype(float)
# Import train and test raw data
train_data_raw = pd.read_csv("train.csv")
test_data_raw = pd.read_csv("test.csv")

train_data = optimize_data(train_data_raw)
test_data = optimize_data(test_data_raw)
#train_data = train_data_raw[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
train_data_results = train_data_raw["Survived"]

# scaling the data
train_data_scaled = scale(train_data)

# with gridsearch
parameters = {'min_samples_split': np.linspace(2,100,99),
              'criterion': ["gini", "entropy"],
              'splitter': ["best","random"]
             }
clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters)
model = clf.fit(train_data_scaled, train_data_results)

# without gridsearch
model2 = tree.ExtraTreeClassifier().fit(train_data_scaled, train_data_results)

res1 = model.predict(train_data_scaled)
res2 = model2.predict(train_data_scaled)

print "gridsearch accuracy: " + str(accuracy_score(train_data_results, res1))
print "regular accuracy: " + str(accuracy_score(train_data_results, res2))


# scaling the data
test_data_scaled = scale(test_data)
#predicting the outcome
test_prediction = model2.predict(test_data_scaled)

output = pd.DataFrame({
    "PassengerId": test_data_raw["PassengerId"],
    "Survived": test_prediction
    })
output = output.set_index("PassengerId")
output.to_csv("output.csv")

