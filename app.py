import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor  # Ensure this matches the model used during training

# Load your trained model
try:
    with open('salary_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Remove or comment out the debug message
    # st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the one-hot encoded columns
try:
    with open('one_hot_columns.pkl', 'rb') as file:
        one_hot_columns = pickle.load(file)
    # Remove or comment out the debug message
    # st.write("One-hot columns loaded successfully.")
except Exception as e:
    st.error(f"Error loading one-hot columns: {e}")

# Define your Streamlit app
st.title('Salary Prediction App')

# Input fields for the user
designation = st.selectbox('Choose your Designation', ['Entry Level', 'Senior', 'Manager', 'Lead', 'Director'])
age = st.number_input('Enter your Age', min_value=18, max_value=70, value=25)
past_exp = st.number_input('Enter your Past Experience (in years)', min_value=0, max_value=50, value=5)

# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame for the inputs with all zeros
    input_data = pd.DataFrame(np.zeros((1, len(one_hot_columns))), columns=one_hot_columns)
    
    # Set the input features based on user inputs
    input_data.at[0, 'AGE'] = age
    input_data.at[0, 'PAST EXP'] = past_exp
    
    # Set the appropriate designation column to 1
    if f'DESIGNATION_{designation}' in input_data.columns:
        input_data.at[0, f'DESIGNATION_{designation}'] = 1
    
    # Remove or comment out the debug message
    # st.write("Input data for prediction:")
    # st.write(input_data)
    
    # Predict the salary
    try:
        salary_prediction = model.predict(input_data)[0]
        # Display the result
        st.write(f'Predicted Salary: ${salary_prediction:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
