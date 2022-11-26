# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:15:24 2022

@author: Iyama
"""


import numpy as np
import pickle



# loading the saved model
model = pickle.load(open('C:/Users/Iyama/Desktop/ML/MLheart/heart_disease_model.sav', 'rb'))

    
# changing the input_data to numpy array
input_data = (38,1,2,138,175,0,1,173,0,0,2,4,2)

input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print('The Test is Positive no heart disease')
else:
    print('The Test is Negative, heart disease predicted')