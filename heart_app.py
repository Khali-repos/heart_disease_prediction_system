# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:10:18 2022

@author: Iyama
"""
import numpy as np
import pickle
import streamlit as st



primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

#loading saved models

loaded_model = pickle.load(open('C:/Users/Iyama/Desktop/ML/MLheart/heart_disease_model.sav','rb'))

def heart(input_data):
     
     

   input_data_as_numpy_array= np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

   prediction = loaded_model.predict(input_data_reshaped)
   print(prediction)

   if (prediction[0]== 0):
       return'The Test is Positive no heart disease'
   else:
       return'The Test is Negative, heart disease predicted'


def main():
     

    
    # page title
    st.title('Heart Disease Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar in mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')
        
    with col1:
        thal = st.text_input('Thalassemia')
        
        
     
     
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        diagnosis = heart([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])                          
        
        
    st.success(diagnosis)
        
        
if __name__ == '__main__':
    main()
    