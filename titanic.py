#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import sklearn

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)
    else:
        return "Number of predictions does not match number of outcomes!"

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    predictions = []
    for _, passenger in data.iterrows():
        if (passenger.Sex == "female") & (passenger.SibSp <= 2):
            predictions.append(1)
        elif (passenger.Age <= 10) & (passenger.SibSp <= 3) & (passenger.Sex == "male"):
            predictions.append(1)
        elif (passenger.Fare <= 100) & (passenger.Fare >= 80):
            predictions.append(1)
        elif (passenger.Fare <= 140) & (passenger.Fare >= 120):
            predictions.append(1)
        elif (passenger.Fare <= 160) & (passenger.Fare >= 180):
            predictions.append(1)
        elif (passenger.Fare >= 500):
            predictions.append(1)

        else:
            predictions.append(0)
    return pd.Series(predictions)

# Import train and test raw data
train_data_raw = pd.read_csv("train.csv")
test_data_raw = pd.read_csv("test.csv")

train_data = train_data_raw[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
train_data_results = train_data_raw["Survived"]

# male   = 1
# female = 0
train_data.ix[train_data.Sex == "male", "Sex"] = 1
train_data.ix[train_data.Sex == "female", "Sex"] = 0
train_data.Age = train_data.fillna(train_data.Age.mean())

from sklearn.preprocessing import scale
# normalizing the data
train_data_scaled = scale(train_data)

from sklearn import svm
clf = svm.SVC(decision_function_shape='poly')
clf.fit(train_data_scaled, train_data_results)
print accuracy_score(train_data_results,clf.predict(train_data_scaled))

from sklearn import grid_search
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf2 = grid_search.GridSearchCV(svm.SVC(), parameters)
clf2.fit(train_data_scaled, train_data_results)
print accuracy_score(train_data_results,clf2.predict(train_data_scaled))

test_data = test_data_raw[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]]



predictions = predictions_3(test_data)
print accuracy_score(train_data_results, predictions)

output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
    })
output = output.set_index("PassengerId")
output.to_csv("output.csv")

