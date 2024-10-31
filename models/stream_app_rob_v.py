
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# load model and test sequences for RUL predictions
model_rul = tf.keras.models.load_model('models\Trained models\Robs_preloaded_lstm_model\model_20241027_212326_lstm.keras') 
#model_rul = tf.keras.models.load_model('../best_rul_model_.keras')     

test_seq_rul = np.load("data\Robs_seq_files\X_test_seq.npy", allow_pickle=True)
test_label_rul = np.load("data\Robs_seq_files\y_test_seq.npy", allow_pickle=True)

# test_seq_rul_1 = np.load("../data/X_test_seq_rul_1.npy", allow_pickle=True)
# test_seq_rul_2 = np.load("../data/X_test_seq_rul_2.npy", allow_pickle=True)
# test_seq_rul_3 = np.load("../data/X_test_seq_rul_3.npy", allow_pickle=True)
# test_seq_rul_4 = np.load("../data/X_test_seq_rul_4.npy", allow_pickle=True)
# test_seq_rul_5 = np.load("../data/X_test_seq_rul_5.npy", allow_pickle=True)
# test_seq_rul = np.vstack((test_seq_rul_1, test_seq_rul_2, test_seq_rul_3, test_seq_rul_4, test_seq_rul_5))
# test_label_rul = np.load("../data/y_test_rul_seq.npy", allow_pickle=True)
# load model and test sequences for telemetry predictions
model_tele = tf.keras.models.load_model('Trained models/GRU_model.keras')     # load telemetry model
test_seq_tele = np.load("../data/test_seq_tele.npy", allow_pickle=True)      # Need to create file for test data sequence
test_seq_tele = tf.convert_to_tensor(test_seq_tele, dtype=tf.float32)
test_label_tele = np.load("../data/test_label_tele.npy", allow_pickle=True)

st.set_page_config(layout="wide")

# Sidebar content
st.sidebar.header("Predictive Maintenance App", divider='blue')
st.sidebar.write("This app predicts RUL for the selected machine.")
st.sidebar.write("Select a machineID in the list below.")
m_id_array = np.arange(1, 101)      # All machineID's (1-100)
m_id_choice = st.sidebar.selectbox("Choose machineID", m_id_array)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("Check below to show predictions of sensor data.")
chk = st.sidebar.checkbox("Sensor data predictions")

# Index for Test data set: machinID 1 has {1464-seq_length} rows, and the other machines have 1464 rows in test data set
rows_rul = 144100
rows_tele = 146400
def data_indices(data, rows):
    seq_length = rows - len(data)
    data_id1 = (rows/100) - seq_length
    if m_id_choice == 1:
        start_index = 0
        end_index = data_id1
    else:
        start_index = ((m_id_choice-2)*(rows/100)) + data_id1
        end_index = start_index + (rows/100)
    return int(start_index), int(end_index), seq_length

start_rul, end_rul, seq_len_rul = data_indices(test_seq_rul, rows_rul)
start_tele, end_tele, seq_len_tele = data_indices(test_seq_tele, rows_tele)

# Set min and max datetime for test data
dt_min = datetime(2015, 11, 1, 0)
dt_max = datetime(2015, 12, 31, 23)

col1, col2 = st.columns([0.5, 0.5])

# RUL Predictions
y_pred_log = model_rul.predict(test_seq_rul[start_rul:end_rul])
y_pred_rul = np.expm1(y_pred_log)

def plot_rul(pred, actual, start, end):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(np.expm1(actual[start:end]), label='Actual RUL')
    plt.plot(pred, label='Predicted RUL', alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('RUL')
    plt.legend()
    plt.title('Predicted vs. Actual RUL (Original Scale)')
    st.pyplot(fig)

with col1:
    st.subheader(f"Predicted RUL for machineID {m_id_choice}", divider='gray')
    pred_rul_hrs = y_pred_log[-1][0]       # Find the last predicted RUL value
    st.subheader(f"RUL is approximately :blue[{pred_rul_hrs:.2f}] hours")      
    st.subheader("")
    plot_rul(y_pred_rul, test_label_rul, start_rul, end_rul)

# Telemetry Predictions
def plot_preds(preds, actuals, start, end, labels):
    fig, axs = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
    fig.tight_layout(pad=1.8)
    axs[0].plot(actuals[start:end, 0], color='green', label='Actual')
    axs[0].plot(preds[start:end, 0], color='red', label='Predicted')
    axs[1].plot(actuals[start:end, 1], color='green', label='Actual')
    axs[1].plot(preds[start:end, 1], color='red', label='Predicted')
    axs[2].plot(actuals[start:end, 2], color='green', label='Actual')
    axs[2].plot(preds[start:end, 2], color='red', label='Predicted')
    axs[3].plot(actuals[start:end, 3], color='green', label='Actual')
    axs[3].plot(preds[start:end, 3], color='red', label='Predicted')
    axs[0].set_title('Volt')
    axs[1].set_title('Rotate')
    axs[2].set_title('Pressure')
    axs[3].set_title('Vibration')
    plt.xticks(np.linspace(0, end-start, 4), labels)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5,-1.0))
    st.pyplot(fig)
    
if chk:
    with col2:
        st.subheader(f"Predicted sensor data for machineID {m_id_choice}", divider='gray')
        pred_tele = model_tele.predict(test_seq_tele[start_tele:end_tele])       
        if m_id_choice == 1:
            min = dt_min + timedelta(hours=seq_len_tele)
        else:
            min = dt_min
        max = dt_max
        start, end = st.slider("Select time range", min_value=min, max_value=max, value=(min, max), format="YYYY-MM-DD HH:mm:ss", step=timedelta(hours=1))
        index_1 = int((start - min).total_seconds()/3600)
        index_2 = int((end-min).total_seconds()/3600)
        labels = pd.date_range(start, end, periods=4)
        plot_preds(pred_tele, test_label_tele[start_tele:end_tele], index_1, index_2, labels)        


