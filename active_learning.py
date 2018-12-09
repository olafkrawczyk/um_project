import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

col_names = ['class', 'alcohol', 'ma', 'ash', 'aoa', 'mg', 'tp', 'fl', 'np', 'pr', 'color', 'hue', 'od', 'proline']
dataset = pd.read_csv('./datasets/wine/wine.data', names=col_names)

y = dataset['class'] # Klasy
x = dataset[col_names[1:]] # Pozostałe cechy

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

clf = SVC(gamma='auto', probability=True)
clf.fit(xTrain, yTrain)

## To co nas interesuje do Active Learing to wsparcia wygenerowane przez klasyfikator
## Dokumentacja https://scikit-learn.org/stable/modules/svm.html

print(clf.support_vectors_)


## Ocena jakości

predictions = clf.predict(xTest)
score = accuracy_score(yTest, predictions)

print(score)

