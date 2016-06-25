#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import grid_search
from sklearn.preprocessing import scale
import numpy as np
import time

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        # print "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)
        return (truth == pred).mean() * 100
    else:
        return "Number of predictions does not match number of outcomes!"

def optimize_data(df_):
    df = df_.copy()
    try:
        df.drop(["PassengerId"], inplace=True, axis=1)
    except: pass
    try:
        df.drop(["Survived"], inplace=True, axis=1)
    except: pass
    df.loc[df.Sex != 'male', 'Sex'] = 0
    df.loc[df.Sex == 'male', 'Sex'] = 1
    df.Age = df.Age.fillna(df.Age.mean())

    df.loc[df.Cabin.str[0] == 'A', 'Cabin'] = 1
    df.loc[df.Cabin.str[0] == 'B', 'Cabin'] = 2
    df.loc[df.Cabin.str[0] == 'C', 'Cabin'] = 3
    df.loc[df.Cabin.str[0] == 'D', 'Cabin'] = 4
    df.loc[df.Cabin.str[0] == 'E', 'Cabin'] = 5
    df.loc[df.Cabin.str[0] == 'F', 'Cabin'] = 6
    df.loc[df.Cabin.str[0] == 'G', 'Cabin'] = 7
    try:
        df.loc[df.Cabin.str[0] == 'T', 'Cabin'] = 8
    except: pass
    df.Cabin.fillna('0', inplace=True)


    df.Embarked.fillna(0, inplace=True)
    df.loc[df.Embarked == 'C', 'Embarked'] = 1
    df.loc[df.Embarked == 'Q', 'Embarked'] = 2
    df.loc[df.Embarked == 'S', 'Embarked'] = 3
    df.loc[(df.Fare > 0) & (df.Fare <= 20), "Fare" ] = 20
    df.loc[(df.Fare > 20) & (df.Fare <= 40), "Fare"] = 40
    df.loc[(df.Fare > 40) & (df.Fare <= 60), "Fare"] = 60
    df.loc[(df.Fare > 60) & (df.Fare <= 80), "Fare"] = 80
    df.loc[(df.Fare > 80) & (df.Fare <= 100), "Fare"] = 100
    df.loc[(df.Fare > 100) & (df.Fare <= 150), "Fare"] = 150
    df.loc[(df.Fare > 150), "Fare"] = 200
    title = df.Name.str.replace("[^aA-zZ \w]", "").str.split().str[1]
    title = title.replace(['Mlle', 'Ms', "Miss"], 1) # 'Miss'
    title = title.replace(['Mme', 'Mrs'], 2) # Mrs
    title = title.replace(['Mr'], 3)  # Mrs
    title = title.replace(['Rev', 'Dr', 'Master', 'Major', 'Col', 'Capt', 'Jonkheer', 'Dona'], 4) # Esteemed
    title = title.replace(['Don', 'Lady', 'Sir', 'the Countess'], 5) # Royalty
    title = pd.to_numeric(title, errors='coerce').fillna(0)
    df.Name = title

    ticket_to_count = dict(df.Ticket.value_counts())
    df['TicketCount'] = df['Ticket'].map(ticket_to_count.get)

    ticket_ID = dict(df.Ticket.str.replace("[^aA-zZ \w 0-9]", "").str.split().str[0].value_counts())
    df["ticket_ID"] = df.Ticket.str.replace("[^aA-zZ \w 0-9]", "").str.split().str[0].map(ticket_ID.get).fillna(0)

    # If data type is categorical, convert to dummy variables
    #df = df.join(pd.get_dummies(df["Pclass"], prefix="Pclass"))
    df = df.join(pd.get_dummies(df["Cabin"], prefix="Cabin"))
    df = df.join(pd.get_dummies(df["Embarked"], prefix="Embarked"))
    df = df.join(pd.get_dummies(df["Fare"], prefix="Fare"))
    # removing the original columns
    #df.drop('Pclass', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('Embarked', axis=1, inplace=True)
    df.drop('Fare', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)

    df.fillna(-1, inplace=True)

    return df.astype(float)

def check_model(model, train_scaled, train_res):
    from sklearn.cross_validation import KFold
    kf = KFold(train_scaled.__len__(), n_folds=10)
    acc = []
    for train_index, test_index in kf:
        # print("TRAIN:", (train_index.min(),train_index.max()), "TEST:", (test_index.min(),test_index.max()))

        model.fit(train_scaled[train_index,:], train_res[train_index])
        prediction = model.predict(train_scaled[test_index,:])
        score = accuracy_score(prediction, train_res[test_index])
        print str(type(model)) + " model accuracy: " + str(score)
        acc.append(score)
    import matplotlib.pyplot as plt
    plt.plot(acc)
    plt.show()
    # model.fit(X_test, y_test)
    # prediction = model.predict(X_train)
    # print "model accuracy: " + str(accuracy_score(prediction, y_train))
    return model

def check_cv(svc, X,y):
    c_len = 10
    gamm_len = 10
    C_s = np.logspace(-1, 4, c_len)
    gamma_s = np.logspace(-4, 1, gamm_len)
    scores = np.zeros([c_len, gamm_len])
    scores_std = np.zeros([c_len, gamm_len])

    # Do the plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(scores)
    fig.colorbar(cax)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    plt.tight_layout()
    i=0
    j=0
    from pylab import draw
    for C in C_s:
        j = 0
        for gamma in gamma_s:
            print "C: {}\t\t Gamma: {}\t".format(np.round(C,5),np.round(gamma,5)),
            svc.C = C
            svc.gamma = gamma
            this_scores = cross_validation.cross_val_score(svc, X, y, n_jobs=1)
            scores[i][j] = np.mean(this_scores)
            scores_std[i][j] = np.std(this_scores)
            print scores[i][j]
            plt.clf()
            cax = ax.matshow(scores)

            j+=1
        i+=1


    pass

# Import train and test raw data
train_data_raw = pd.read_csv("train.csv")
test_data_raw = pd.read_csv("test.csv")

train_data = optimize_data(train_data_raw)
test_data = optimize_data(test_data_raw)

# Remove the features with low variance
from sklearn.feature_selection import VarianceThreshold
P = .8
sel = VarianceThreshold(threshold=(P * (1 - P)))
train_data = train_data[train_data.columns[sel.fit(train_data).variances_>(P * (1 - P))]]
test_data = test_data[train_data.columns[sel.fit(train_data).variances_>(P * (1 - P))]]

# test_data must contain the same columns as train data due to model fitting and prediction
for column in train_data.columns:
    if not column in test_data.columns:
        test_data[column] = pd.DataFrame().apply(lambda _: '', axis=1)
test_data.fillna(0, inplace=True)

#train_data = train_data_raw[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
train_data_results = train_data_raw["Survived"]

# scaling train data
train_data_scaled = scale(train_data)
# scaling test data
test_data_scaled = scale(test_data)


# with gridsearch
# parameters = {'min_samples_split': np.linspace(2,100,49),
#               'criterion': ["gini", "entropy"],
#               'splitter': ["best","random"],
#              }
# clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters,cv = 5)
#model = clf.fit(train_data_scaled, train_data_results)
# without gridsearch
check_cv(svm.SVC(), train_data_scaled, train_data_results)
model = check_model(svm.SVC(), train_data_scaled, train_data_results)
from sklearn.learning_curve import learning_curve
a, on_train, on_test= learning_curve(svm.SVC(), train_data_scaled, train_data_results)
#predicting the outcome
test_prediction = model.predict(test_data_scaled)

output = pd.DataFrame({
    "PassengerId": test_data_raw["PassengerId"],
    "Survived": test_prediction
    })
output = output.set_index("PassengerId")
output.to_csv("output.csv")

