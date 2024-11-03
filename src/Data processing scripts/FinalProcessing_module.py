import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Logging_module import LoggerSetup


class FinalProcessing:
    """
    A class to handle data scaling, splitting, and sequence creation.
    """

    # Feature columns and time steps set as class variables (fixed)
    FEATURE_COLUMNS = [
        'machineID', 'age', '_model1', '_model2', '_model3', '_model4',
        'time_since_last_maint', 'time_since_last_error', 'rotate', 'pressure', 
        'vibration', 'volt', 'failure_indicator', 'maint_indicator', 'error_indicator'
    ]
    TIME_STEPS = 24

    def __init__(self, logger_name='scaling_sequence_logger', log_file='scaling_sequence_log.log'):
        """
        Initializes the FinalProcessing class with a logger.

        Args:
            logger_name (str): The name of the logger.
            log_file (str): The file where the log output will be saved.
        """
        logger_setup = LoggerSetup(logger_name=logger_name, log_file=log_file)
        self.logger = logger_setup.get_logger()
        self.scalers = {}

    def _split_data(self, df):
        """
        Splits data into expanding training, rolling validation, and test sets.

        Args:
            df (DataFrame): The DataFrame to be split.

        Returns:
            DataFrame: Training, validation, and test DataFrames.
        """
        
        df = df.sort_values(by=['machineID', 'datetime'])

        
        latest_date = df['datetime'].max()
        training_end = latest_date - pd.DateOffset(months=6)  
        validation_end = latest_date - pd.DateOffset(months=2)  
        test_end = latest_date  

        # Define masks for time-based splits
        train_mask = (df['datetime'] <= training_end) 
        val_mask = (df['datetime'] > training_end) & (df['datetime'] <= validation_end)
        test_mask = (df['datetime'] > validation_end) & (df['datetime'] <= test_end)

        # Split into train, validation, and test sets
        df_train = df[train_mask]
        df_val = df[val_mask]
        df_test = df[test_mask]

        self.logger.info("Data split into expanding training set, rolling validation, and test sets successfully.")
        return df_train, df_val, df_test

    def _fit_scalers(self, df_train):
        """
        Fits MinMaxScalers for the sensor data.

        Args:
            df_train (DataFrame): The training DataFrame to fit the scalers.
        """
        features_to_scale = ['volt', 'rotate', 'pressure', 'vibration', 'time_since_last_error', 'time_since_last_maint']
        
        for feature in features_to_scale:
            self.scalers[feature] = MinMaxScaler().fit(df_train[[feature]])
            self.logger.info(f"Scaler fitted for feature: {feature}")

    def _scale_data(self, df):
        """
        Scales sensor data using fitted scalers.

        Args:
            df (DataFrame): The DataFrame to be scaled.

        Returns:
            DataFrame: The scaled DataFrame.
        """
        try:
            for feature, scaler in self.scalers.items():
                df[feature] = scaler.transform(df[[feature]])
                self.logger.info(f"Feature '{feature}' scaled successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error while scaling data: {e}")
            raise

    def _apply_log_transform(self, df):
        """
        Applies log transformation to the RUL feature.

        Args:
            df (DataFrame): The DataFrame containing the 'RUL' column.

        Returns:
            DataFrame: The DataFrame with 'RUL_log' column.
        """
        try:
            df['RUL_log'] = np.log1p(df['RUL'])
            self.logger.info("Log transformation applied to 'RUL'.")
            return df
        except Exception as e:
            self.logger.error(f"Error while applying log transformation: {e}")
            raise

    def _create_sequences(self, X, y, time_steps):
        """
        Creates sequences for LSTM training.

        Args:
            X (numpy array): Feature array.
            y (numpy array): Target variable.
            time_steps (int): Number of time steps for each sequence.

        Returns:
            numpy array: Sequences for features and target.
        """
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])

        self.logger.info(f"Sequences created with time steps: {time_steps}")
        return np.array(Xs), np.array(ys)

    def split_scale_sequence(self, df):
        """
        Performs scaling, applies log transform, and creates sequences for LSTM.

        Args:
            df (DataFrame): The DataFrame to be processed.

        Returns:
            numpy arrays: Scaled sequences for train, validation, and test sets.
        """
        # Split the data
        df_train, df_val, df_test = self._split_data(df)

        # Fit scalers on training set and apply scaling
        self._fit_scalers(df_train)
        df_train = self._scale_data(df_train)
        df_val = self._scale_data(df_val)
        df_test = self._scale_data(df_test)

        # Apply log transform to RUL
        df_train = self._apply_log_transform(df_train)
        df_val = self._apply_log_transform(df_val)
        df_test = self._apply_log_transform(df_test)

        # Extract features and target variable
        X_train = df_train[self.FEATURE_COLUMNS].values
        y_train = df_train['RUL_log'].values

        X_val = df_val[self.FEATURE_COLUMNS].values
        y_val = df_val['RUL_log'].values

        X_test = df_test[self.FEATURE_COLUMNS].values
        y_test = df_test['RUL_log'].values

        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, self.TIME_STEPS)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, self.TIME_STEPS)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, self.TIME_STEPS)

        return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
