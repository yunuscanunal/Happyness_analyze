#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import LocalOutlierFactor 
from sklearn import preprocessing


happyness = pd.read_csv("ACME-HappinessSurvey2020.csv")
data1 = happyness.copy()


data1.head()

data1.info()

data1.describe().T

data1.isnull().sum()

sns.heatmap(data1.corr(), annot=True, cmap= "BuPu");

from collections import Counter


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


detect_outliers(data1, data1.columns)

sns.boxplot(data1["X1"]);


clf= LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(data1)

data_scores=clf.negative_outlier_factor_
np.sort(data_scores)[0:10]


sns.boxplot(data_scores);


outlier_indexes=data1.loc[data_scores< -1.68405746]
outlier_indexes


y_data1 = data1["Y"]
y_data1.isnull().sum()
y_data1=pd.DataFrame(y_data1).drop(index=[119])
y_data1=y_data1.reset_index(drop=True)
print(y_data1.shape)
y_data1.head(2)
y_data1.isnull().sum()

print(y_data1.shape)
data1 = data1.drop(axis= 1, columns=["Y"])

data1=pd.DataFrame(data1).drop(index=[119])
data1=data1.reset_index(drop=True)
print(data1.shape)



data2 = happyness.copy()

outlier_indexes=data2.loc[data_scores< -1.31670736]
outlier_indexes

y_data2 = data2["Y"]
y_data2.isnull().sum()
y_data2=pd.DataFrame(y_data2).drop(index=[6,9,34,47,56,63,71,94,116,119])
y_data2=y_data2.reset_index(drop=True)
print(y_data2.shape)
y_data2.head(2)

data2 = data2.drop(index= [6,9,34,47,56,63,71,94,116,119])
data2 = data2.reset_index(drop=True)
print(data2.shape)

data2 = data2.drop(axis= 1, columns=["Y"])
print(data2.shape)

data2.shape

data2.info()



data3 = happyness.copy()

from sklearn.metrics import mean_squared_error

loj = sm.Logit(y, X)
loj_model = loj.fit()
loj_model.summary()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)
loj_model

loj_model.intercept_


loj_model.coef_

y_pred = loj_model.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)


y=happyness["Y"]
X=happyness.drop("Y", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)
loj_model

y_pred = loj_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))


y=y_data1["Y"]
X=data1
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)
loj_model

y_pred = loj_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))

y=y_data2["Y"]
X=data2
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)
loj_model

y_pred = loj_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


svm_model = SVC(kernel = "linear").fit(X_train, y_train)


y=happyness["Y"]
X=happyness.drop("Y", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
svm_model = SVC(kernel = "linear").fit(X_train, y_train)

svm_model

y_pred = svm_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))


y=y_data1["Y"]
X=data1
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
svm_model = SVC(kernel = "linear").fit(X_train, y_train)

svm_model

y_pred = svm_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))


y=y_data2["Y"]
X=data2
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
svm_model = SVC(kernel = "linear").fit(X_train, y_train)

svm_model

y_pred = svm_model.predict(X)
print("acc= ", accuracy_score(y, y_pred))

