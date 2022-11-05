import pickle
import streamlit as st
import numpy as np

model  = pickle.load(open('pipe.pkl','rb'))


st.title('Heart Disease Prediction')


col1, col2 = st.columns(2)
    
with col1:
    
    age = st.text_input('Age')
        
with col2:
    sex = st.text_input('Sex M-1 F-0')
        
with col1:
    cp = st.text_input('Chest Pain types (0-3)')
        
with col2:
    trestbps = st.text_input('Resting Blood Pressure')
        
with col1:
    chol = st.text_input('Serum Cholestoral in mg/dl')
        
with col2:
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0-1)')
        
with col1:
    restecg = st.text_input('Resting Electrocardiographic results (0-2)')
        
with col2:
    thalach = st.text_input('Maximum Heart Rate achieved')
        
with col1:
    exang = st.text_input('Exercise Induced Angina (0-1)')
        
with col2:
    oldpeak = st.text_input('ST depression induced by exercise')
        
with col1:
    slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    
with col2:
    ca = st.text_input('Major vessels colored by flourosopy (0-4)')
        
with col1:
    thal = st.text_input('Thal (0-3)')

try:
    # code for Prediction
    heart_diagnosis = ''
        
    # creating a button for Prediction
    st.button('Heart Disease Test Result')


    heart_prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
            
    if (heart_prediction[0] == 1):
       heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not have any heart disease'
            
    st.success(heart_diagnosis)
except:pass