import streamlit as st
import numpy as np

# Users input the sensor parameters and click the Predict button, 
# they then receive feedback indicating the likelihood of failure and the remaining useful life for each component, 
# helping them make informed maintenance decisions.

st.title('Predictive Maintenance App')

# Load trained model
model = tf.keras.models.load_model('best model path')

# Input fields for user to enter shit
voltage = st.number_input('Enter voltage:', value=0.0)
pressure = st.number_input('Enter pressure:', value=0.0)
rotation = st.number_input('Enter rotation:', value=0.0)
vibration = st.number_input('Enter vibration:', value=0.0)

# Predict based on user inputs
if st.button('Predict'):
    inputs = np.array([[voltage, pressure, rotation, vibration]]) 
    predictions = model.predict(inputs)

    predicted_class = predictions[0][0]  
    predicted_rul = predictions[1]  

    # Display the predictions
    st.write(f'Predicted Failure Class: {"Failure" if predicted_class >= 0.5 else "No Failure"}')
    st.write('Predicted Remaining Useful Life (RUL):')
    st.write(f'Component 1: {predicted_rul[0]:.2f} hours')
    st.write(f'Component 2: {predicted_rul[1]:.2f} hours')
    st.write(f'Component 3: {predicted_rul[2]:.2f} hours')
    st.write(f'Component 4: {predicted_rul[3]:.2f} hours')