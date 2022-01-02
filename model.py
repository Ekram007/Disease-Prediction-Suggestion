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

    dengue_doctor = {"Cirrhosis": "Dr. Haridas Saha ,Hepatologist", 
                    "Diabetes Mellitus": "Dr. Indrajit Prasad , Endocrinologist", 
                    "Thyroid":"Endocrinologist, Dr. Indrajit Prasad , " ,
                    "Glomeroulonephritis":"Prof. Dr. Asia Khanam, Nephrologist", 
                    "Carcinoma": "Dr. AFM Kamal Uddin , Oncologist", 
                    "Pregnancy": "Dr. Begum Hosne Ara, Gynecologist",
                    "Arrythmia":"Dr. AKS Zahid Mahmud Khan , Cardiologist", 
                    "Bone Pain":"Dr. Abu Shahin , Rheumatologist", 
                    "Measles":"Prof. Dr. Md. Habibur Rahman, Infectious Disease Specialist", 
                    "Pulmonary Disease":"Dr. Md. Sajedul Islam, Pulmonologist" }

    dengue_medicine = {"Cirrhosis": "Paracetamol, Lactulose, Drink water.", 
                        "Diabetes Mellitus": "Paracetamol, Metformin, Drink water.", 
                        "Thyroid":"Paracetamol, Thyroxin, Drink water." ,
                        "Glomeroulonephritis":"Paracetamol, ACE Inhibitors, Drink water.", 
                        "Carcinoma": "Paracetamol, Chemotherapy, Drink water.", 
                        "Pregnancy": "Paracetamol, Drink water.",
                        "Arrythmia":"Paracetamol, Encainide, Drink water.", 
                        "Bone Pain":"Paracetamol, Tylenol, Drink water.", 
                        "Measles":"Paracetamol, Vitamain-A, Measles virus vaccine, Drink water.", 
                        "Pulmonary Disease":"Paracetamol, Salmeterol Inhaler, Drink water." }

    return dengue_doctor[prev_history] , dengue_medicine[prev_history]

def thyroid_doc_med(prev_history):

    thyroid_doctor = {"Chicken Pox": "Dr. Tasnim Khan, Dermatologist",
                        "Common Cold": "Dr. Md. Rezaul Karim, General Medicine", 
                        "Periodontitis":"Dr. Shahriar Karim Sajid, Dentist" ,
                        "Dengue":"Dr. Shafat Hossain, Critical Care Specialist", 
                        "Diabetic Retinopathy": "Dr. Shomir Hossain, Opthalmologist", 
                        "Hay Fever": "Dr. Tareq Alam, Immunologist",
                        "Schizophrenia":"Dr. Hasan-Al-Mamun, Psychiatrist", 
                        "PCOS":"Dr. Begum Hosne Ara, Gynecologist", 
                        "Erectile dysfunction":"Dr. Md. Jahangir Kabir, Urologist", 
                        "Bell's Palsy":"Dr. AF Mohiuddin Khan, ENT Specialist" }

    thyroid_medicine = {"Chicken Pox": "Acyclovir, regular exercise.", 
                        "Common Cold": "Fexofenadine, regular exercise.", 
                        "Periodontitis":"Doxycycline, regular exercise." ,
                        "Dengue":"Paracetamol, regular exercise.", 
                        "Diabetic Retinopathy": "Ranibizumab, regular exercise.", 
                        "Hay Fever": "Claritin, regular exercise.",
                        "Schizophrenia":"Procyclidine HCl, regular exercise.", 
                        "PCOS":"Metformin, regular exercise.", 
                        "Erectile dysfunction":"Sildenafil, regular exercise.", 
                        "Bell's Palsy":"Prednisolone, regular exercise." }

    return thyroid_doctor[prev_history], thyroid_medicine[prev_history]

def diabetes_doc_med(prev_history):

    diabetes_doctor = {"COPD": "Prof. Dr. Asia Khanam, Nephrologist", 
                        "Peripheral Vascular": "Dr. AKS Zahid Mahmud Khan, Cardiologist", 
                        "Dislipidemia":"Dr. AKS Zahid Mahmud Khan, Cardiologist" ,
                        "Neuropathy":"Dr. Bahadur Ali Miah, Neurologist", 
                        "Psoriasis": "Dr. Tasnim Khan, Dermatologist", 
                        "Acne": "Dr. Tasnim Khan, Dermatologist",
                        "Pulpitis":"Dr. Shahriar Karim Sajid, Dentist", 
                        "Glaucoma":"Dr. Shomir Hossain, Opthalmologist", 
                        "Thyroid":"Dr. Indrajit Prasad, Endocrinologist", 
                        "Coeliac":"Dr. Chanchal Kumar Ghosh, Gastroenterologist" }

    diabetes_medicine = {"COPD": "Roflumilast, Diet Control & Exercise.", 
                        "Peripheral Vascular": "Anti-clotting agents, Diet Control & Exercise.", 
                        "Dislipidemia":"Fibrates and Statins, Diet Control & Exercise. " ,
                        "Neuropathy":"Amitriptyline, Diet Control & Exercise.", 
                        "Psoriasis": "Cerave Psoriasis Cream, Diet Control & Exercise.", 
                        "Acne": "Dapsone, Diet Control & Exercise.",
                        "Pulpitis":"NSAID's, Acetominophen, Diet Control & Exercise.", 
                        "Glaucoma":"Bimatoprost drop, Diet Control & Exercise.", 
                        "Thyroid":"Thyroxin, Diet Control & Exercise.", 
                        "Coeliac":"Gluten Free Diet, Diet Control & Exercise." }

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
