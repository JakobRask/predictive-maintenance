import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
# import LSTM????

st.sidebar.header("Predicting RUL for machines.", divider='blue')
st.sidebar.write(
    "This app predicts RUL for the selected machine." 
    )
st.sidebar.write(
    "Select a machineID in the list below."
    )

# List with machineID's
m_id_array = np.arange(1, 101)
m_id = st.sidebar.selectbox("Choose machineID", m_id_array)

# Checkbox to see chart
st.sidebar.subheader("Check to show chart")
chk = st.sidebar.checkbox("Chart")

# load model + test set and predict
model = load_model('model.keras')
RUL_test = pd.read_csv("../../data/test_data.csv")      # Need to create csv file for test data set
pred_time, pred_comp = model.predict(RUL_test.loc[RUL_test['machineID'] == m_id])       # Change to whatever output we get from the model

# Prediction text
st.title("Predicted RUL for machineID:{m_id}")
col1, col2, col3, col4 = st.columns(4, gap="small")
with cols1:
    st.subheader(f"{pred_comp[0]}:")
    st.subheader(f":blue[{pred_time[0]}]")

with cols2:
    st.subheader(f"{pred_comp[1]}:")
    st.subheader(f":blue[{pred_time[1]}]")

with cols3:
    st.subheader(f"{pred_comp[2]}:")
    st.subheader(f":blue[{pred_time[2]}]")
    
with cols4:
    st.subheader(f"{pred_comp[3]}:")
    st.subheader(f":blue[{pred_time[3]}]")

# If checkbox is checked, show a chart
with chk:
    chart_data = pd.DataFrame(pred_time, pred_comp, columns=["Hours", "Component"])
    st.bar_chart(chart_data, horizontal=True, width=20, height=10)



