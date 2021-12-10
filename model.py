from keras.models import load_model
import numpy

from app import disease



def disease_prediction(input_data):

    model = load_model('model.h5')

    input_data = numpy.array(input_data)
    input_data = input_data.reshape((1,-1))

    y_pred = model.predict(input_data)
    
    #y_pred=(y_pred>0.5)
    y_pred = numpy.argmax(y_pred)

    disease_mapping = {0 : "Not Diabetes", 1: "Diabetes", 2 : "Not Thyroid", 3 : "Thyroid", 4 : "Not Dengue", 5 : "Dengue"}

    

    return disease_mapping[y_pred]

#disease_prediction([13,145,19,110,22.2,57,82,0.245,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])