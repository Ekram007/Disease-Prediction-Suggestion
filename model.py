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

def dengue_doc_med(prev_history):

    dengue_doctor = {"Cirrhosis": "Hepatologist", "Diabetes Mellitus": "Endocrinologist", "Thyroid":"Endocrinologist" ,"Glomeroulonephritis":"Nephrologist", "Carcinoma": "Oncologist", "Pregnancy": "Gynecologist",
                 "Arrythmia":"Cardiologist", "Bone Pain":"Rheumatologist", "Measles":"Infectious Disease Specialist", "Pulmonary Disease":"Pulmonologist" }

    dengue_medicine = {"Cirrhosis": "Paracetamol, Lactulose", "Diabetes Mellitus": "Paracetamol, Metformin", "Thyroid":"Paracetamol, Thyroxin " ,"Glomeroulonephritis":"Paracetamol, ACE Inhibitors", "Carcinoma": "Paracetamol, Chemotherapy", "Pregnancy": "Paracetamol",
                 "Arrythmia":"Paracetamol, Encainide", "Bone Pain":"Paracetamol, Tylenol", "Measles":"Paracetamol, Vitamain-A, Measles virus vaccine", "Pulmonary Disease":"Paracetamol, Salmeterol Inhaler" }

    return dengue_doctor[prev_history] , dengue_medicine[prev_history]

def thyroid_doc_med(prev_history):

    thyroid_doctor = {"Chicken Pox": "Dermatologist", "Common Cold": "General Medicine", "Periodontitis":"Dentist" ,"Dengue":"Critical Care Specialist", "Diabetic Retinopathy": "Opthalmologist", "Hay Fever": "Immunologist",
                 "Schizophrenia":"Psychiatrist", "PCOS":"Gynecologist", "Erectile dysfunction":"Urologist", "Bell's Palsy":"ENT Specialist" }

    thyroid_medicine = {"Chicken Pox": "Acyclovir", "Common Cold": "Fexofenadine", "Periodontitis":"Doxycycline" ,"Dengue":"Paracetamol", "Diabetic Retinopathy": "Ranibizumab", "Hay Fever": "Claritin",
                 "Schizophrenia":"Procyclidine HCl", "PCOS":"Metformin", "Erectile dysfunction":"Sildenafil", "Bell's Palsy":"Prednisolone" }

    return thyroid_doctor[prev_history], thyroid_medicine[prev_history]

def diabetes_doc_med(prev_history):

    diabetes_doctor = {"COPD": "Nephrologist", "Peripheral Vascular": "Cardiologist", "Dislipidemia":"Cardiologist" ,"Neuropathy":"Neurologist", "Psoriasis": "Dermatologist", "Acne": "Dermatologist",
                 "Pulpitis":"Dentist", "Glaucoma":"Opthalmologist", "Thyroid":"Endocrinologist", "Coeliac":"Gastroenterologist" }

    diabetes_medicine = {"COPD": "Roflumilast", "Peripheral Vascular": "Anti-clotting agents", "Dislipidemia":"Fibrates and Statins " ,"Neuropathy":"Amitriptyline", "Psoriasis": "Cerave Psoriasis Cream", "Acne": "Dapsone",
                 "Pulpitis":"NSAID's, Acetominophen", "Glaucoma":"Bimatoprost drop", "Thyroid":"Thyroxin", "Coeliac":"Gluten Free Diet" }

    return diabetes_doctor[prev_history], diabetes_medicine[prev_history]

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
