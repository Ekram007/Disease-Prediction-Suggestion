import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

dataset = pd.read_csv('dengue.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

MLP = MLPClassifier()
MLP.fit(X_train, y_train)

joblib.dump(MLP, "Trained_Dengue.sav")




"""a = [[13,43,37.5,219,1,0,0]]
model = joblib.load("Trained_Dengue.sav")
result = model.score(X_test, y_test)
print(result)
print(model.predict(a))"""

