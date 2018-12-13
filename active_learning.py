#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

col_names = ['class', 'alcohol', 'ma', 'ash', 'aoa', 'mg',
             'tp', 'fl', 'np', 'pr', 'color', 'hue', 'od', 'proline']
dataset = pd.read_csv('./datasets/wine/wine.data', names=col_names)



y = dataset['class']  # Klasy
x = dataset[col_names[1:]]  # Pozostałe cechy

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.97)

clf = SVC(kernel='linear', gamma='auto', probability=True, C=10000.0)
clf.fit(xTrain, yTrain)

# To co nas interesuje do Active Learing to wsparcia wygenerowane przez klasyfikator
# Dokumentacja https://scikit-learn.org/stable/modules/svm.html

# Ocena jakości

predictions = clf.predict(xTest)
score = accuracy_score(yTest, predictions)

## Accuracy przed AL
print(score)

# 3. Margin selection — we select ‘k’ samples with the lowest difference between the two highest class probabilities, i.e., a higher figure will be given for samples whose model was very certain about a single class and lower to samples whose class probabilities are very similar.

# In principle, active learning can be performed with any classification algorithm that is capable of providing uncer- tainty estimates with class predictions for new samples.

# Zbieramy prawdopodobieństwa klasyfikacji zbioru testowego
df = pd.DataFrame(clf.predict_proba(xTest))
df.index = xTest.index

# Wybieramy największe dla kadej klasy
df['max_value'] = df.max(axis=1)

# Sortujemy majejąco
sorted_df = df.sort_values(by='max_value')

# Wybieramy ~20% najgorszych i wrzucamy je do zbioru uczącego
indexes = sorted_df[:30].index.values
yTrain_AL = pd.DataFrame(yTest).loc[indexes]
xTrain_AL = pd.DataFrame(xTest).loc[indexes]

xTest = xTest.drop(indexes)
yTest = yTest.drop(indexes)

xTrain = pd.concat([xTrain, xTrain_AL])
yTrain = np.concatenate([yTrain, yTrain_AL['class']])

# Ponownie uczymy klasyfikator
clf.fit(xTrain, yTrain)

predictions = clf.predict(xTest)
score = accuracy_score(yTest, predictions)

# Wynik po uczeniu z obiektami najtrudniejszymi w klasyfikacji
#
print(score)
