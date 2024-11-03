import numpy as np
import pandas as pd
from datetime import datetime
import os

from Logging_module import LoggerSetup
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt


class LSTMModelTrainer:
    def __init__(self, time_steps=24, features=15, batch_size=256, epochs=50, model_path='models'):
        self.logger = LoggerSetup(logger_name='LSTMModel_Trainer', log_file='LSTMModel_Trainer.log').get_logger()
        self.logger.info("Initializing LSTMModelTrainer.")

        self.time_steps = time_steps
        self.features = features
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)  # Ensure the model path directory exists
        self.model = self._build_model()
        self.history = None

  

    def _build_model(self):
        input_layer = Input(shape=(self.time_steps, self.features))
        lstm_out = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(input_layer)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(50, activation='tanh', return_sequences=True)(lstm_out)
        lstm_out = LSTM(50, activation='tanh', return_sequences=False)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        rul_pred_log = Dense(64, activation='relu')(lstm_out)
        when_output = Dense(1, name='RUL_log')(rul_pred_log)
        model = Model(inputs=input_layer, outputs=when_output)
        self.logger.info("Model architecture built successfully.")
        return model

    def _compile_model(self):
        huber_loss = Huber()
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae'])
        self.logger.info("Model compiled with Huber loss and Adam optimizer.")
        
    def _get_callbacks(self):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_save_path = os.path.join(self.model_path, f'model_{timestamp}_lstm.keras')
        
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
        
        self.logger.info("Callbacks with timestamped model path created successfully.")
        return [early_stopping, checkpoint, reduce_lr]

    def train(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
        self._compile_model()
        callbacks = self._get_callbacks()
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )
        self.logger.info("Model training complete.")

    def evaluate(self, X_test_seq, y_test_seq):
        results = self.model.evaluate(X_test_seq, y_test_seq)
        test_loss, test_mae = results
        self.logger.info(f"Test Loss for RUL: {test_loss}")
        self.logger.info(f"MAE for RUL: {test_mae}")
        return test_loss, test_mae

    def predict(self, X_test_seq):
        y_pred_log = self.model.predict(X_test_seq)
        y_pred_rul = np.expm1(y_pred_log)
        self.logger.info("Prediction complete.")
        return y_pred_rul

    def plot_training_history(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def plot_rul_predictions(self, y_test_seq, y_pred_rul):
        plt.figure(figsize=(14, 8))
        plt.plot(np.expm1(y_test_seq), label='Actual RUL')
        plt.plot(y_pred_rul, label='Predicted RUL', alpha=0.3)
        plt.xlabel('Time Steps')
        plt.ylabel('RUL')
        plt.legend()
        plt.title('Predicted vs. Actual RUL (Original Scale)')
        plt.show()

    def plot_machine_rul(self, X_test_seq, y_test_seq, y_pred_rul, machine_id):
        machine_test_seq = X_test_seq[:, 0, 0]  # Assuming machineID is the first feature
        rul_comparison_df = pd.DataFrame({
            'machineID': machine_test_seq.flatten(),
            'Actual RUL': np.expm1(y_test_seq).flatten(),
            'Predicted RUL': y_pred_rul.flatten()
        })
        machine_data = rul_comparison_df[rul_comparison_df['machineID'] == machine_id]
        
        plt.figure(figsize=(10, 6))
        plt.plot(machine_data['Actual RUL'].values, label='Actual RUL for Machine')
        plt.plot(machine_data['Predicted RUL'].values, label='Predicted RUL for Machine', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('RUL')
        plt.legend()
        plt.title(f'Predicted vs. Actual RUL for Machine {machine_id}')
        plt.show()
