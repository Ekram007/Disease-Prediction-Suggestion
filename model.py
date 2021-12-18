from keras.models import load_model
import numpy
import joblib
from sklearn.preprocessing import StandardScaler


def dengue_prediction(input_data):

    input_data = numpy.array(input_data)
    input_data = input_data.reshape(1,-1)

    model = joblib.load('Trained_Dengue.sav')
    result = model.predict(input_data)
    dengue_mapping = {0:"Not Dengue", 1:"Dengue"}

    return dengue_mapping[result[0]]

def diabetes_prediction(input_data):

    input_data = numpy.array(input_data)
    input_data = input_data.reshape(1,-1)

    model = joblib.load('Trained_Diabetes.sav')
    result = model.predict(input_data)
    diabetes_mapping = {0:"Not Diabetes", 1:"Diabetes"}

    return diabetes_mapping[result[0]]

def thyroid_prediction(input_data):

    input_data = numpy.array(input_data)
    input_data = input_data.reshape(1,-1)

    model = joblib.load('Trained_Thyroid.sav')
    result = model.predict(input_data)
    thyroid_mapping = {0:"Not Thyroid", 1:"Thyroid"}

    return thyroid_mapping[result[0]]


def disease_prediction(input_data):

    input_data = numpy.array(input_data)
    input_data = input_data.reshape((1,-1))
    
    model = load_model('model.h5')
    y_pred = model.predict(input_data)
    
    #y_pred=(y_pred>0.5)
    y_pred = numpy.argmax(y_pred)

    disease_mapping = {0 : "Not Diabetes", 1: "Diabetes", 2 : "Not Thyroid", 3 : "Thyroid", 4 : "Not Dengue", 5 : "Dengue"}

    return disease_mapping[y_pred]
    
    

"""a= numpy.array([5,6])

a = a.reshape(1,-1)
print(a)"""

#disease_prediction([13,145,19,110,22.2,57,82,0.245,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print(dengue_prediction([[4.94,32,46.63,92,0,0,1]]))
