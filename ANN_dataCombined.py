# At first we have to import the necessary packages from numpy(for managing the arrays), pandas(for reading and writing data), sklearn (for different ML models and ML related library), and keras(for the neural network)
import numpy as np
import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def train():

    dataset = pd.read_csv('dataCombined.csv')

    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    dummy_y = np_utils.to_categorical(encoded_Y)

    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = Sequential()
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 42))
    classifier.add(Dropout(0.1))

    classifier.add(Dense(units =32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.1))

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.1))

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

    classifier.fit(X_train, y_train, batch_size=20, epochs =300)


    classifier.save("model.h5")

train()





