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


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Import train and test raw data
train_data_raw = pd.read_csv("train.csv")
test_data_raw = pd.read_csv("test.csv")

train_data = train_data_raw[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
train_data_results = train_data_raw["Survived"]

# male   = 1
# female = 0
train_data.ix[train_data.Sex == "male", "Sex"] = 1
train_data.ix[train_data.Sex == "female", "Sex"] = 0
train_data.Age = train_data.Age.fillna(train_data.Age.mean())


# normalizing the data
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


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(res1, clf.predict(train_data_scaled))
plot_confusion_matrix(cm)

cm = confusion_matrix(res2, clf.predict(train_data_scaled))
plot_confusion_matrix(cm)


test_data = test_data_raw[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
# male   = 1
# female = 0
test_data.ix[test_data.Sex == "male", "Sex"] = 1
test_data.ix[test_data.Sex == "female", "Sex"] = 0
test_data.Age = test_data.Age.fillna(test_data.Age.mean())
test_data.Fare = test_data["Fare"].fillna(test_data["Fare"].mean())
# normalizing the data
test_data_scaled = scale(test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]])
#predicting the outcome
test_prediction = model2.predict(test_data_scaled)

output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_prediction
    })
output = output.set_index("PassengerId")
output.to_csv("output.csv")

