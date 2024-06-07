import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor 


try:
    with open('salary_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
except Exception as e:
    st.error(f"Error loading model: {e}")


try:
    with open('one_hot_columns.pkl', 'rb') as file:
        one_hot_columns = pickle.load(file)
    
except Exception as e:
    st.error(f"Error loading one-hot columns: {e}")


st.title('Salary Prediction App')


designation = st.selectbox('Choose your Designation', ['Entry Level', 'Senior', 'Manager', 'Lead', 'Director'])
age = st.number_input('Enter your Age', min_value=18, max_value=70, value=25)
past_exp = st.number_input('Enter your Past Experience (in years)', min_value=0, max_value=50, value=5)


if st.button('Predict Salary'):
    
    input_data = pd.DataFrame(np.zeros((1, len(one_hot_columns))), columns=one_hot_columns)
    
    
    input_data.at[0, 'AGE'] = age
    input_data.at[0, 'PAST EXP'] = past_exp
    
    
    if f'DESIGNATION_{designation}' in input_data.columns:
        input_data.at[0, f'DESIGNATION_{designation}'] = 1
    
    
    try:
        salary_prediction = model.predict(input_data)[0]
        
        st.write(f'Predicted Salary: ${salary_prediction:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
