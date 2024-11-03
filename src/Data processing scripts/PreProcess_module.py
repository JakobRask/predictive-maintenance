import pandas as pd
from Logging_module import LoggerSetup


class PreProcessor:
    """
    A class to perform data preprocessing, merging, and cleaning on DataFrames.
    """
    
    def __init__(self, dataframes, logger_name='data_preprocesser_logger', log_file='data_preprocessor_log.log'):
        """
        Initializes the PreProcessor with a dictionary of DataFrames and sets up a log to track data transformations.

        Args:
            dataframes (dict): A dictionary containing multiple DataFrames to be processed.
            logger_name (str): The name of the logger.
            log_file (str): The file where the log output will be saved.
        """
        self.dataframes = dataframes
        logger_setup = LoggerSetup(logger_name=logger_name, log_file=log_file)
        self.logger = logger_setup.get_logger()

    def _remove_duplicates(self, df):
        """
        Removes duplicate rows from the DataFrame.
        """
        original_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        duplicates_removed = original_rows - final_rows
        self.logger.info(f"Removed {duplicates_removed} duplicates. Final number of rows: {final_rows}")
        return df

    def _change_column_dtypes(self, df):
        """
        Changes data types of the columns based on their content:
        
        - Integer-like columns are converted to int.
        - Decimal-like columns are converted to float.
        - Datetime is kept intact for further processing.
        - All other columns are converted to str.
        """
        for column in df.columns:
            if column == 'datetime':
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df['datetime']):
                        self.logger.info(f"Successfully converted 'datetime' column to datetime type.")
                    else:
                        self.logger.warning(f"Column 'datetime' could not be converted to datetime.")
                except Exception as e:
                    self.logger.error(f"Failed to convert column 'datetime' to datetime data type. Error: {e}")
            else:
                try:
                    numeric_col = pd.to_numeric(df[column], errors='coerce')
                    if numeric_col.notna().all():
                        if (numeric_col % 1 == 0).all():
                            df[column] = numeric_col.astype(int)
                            self.logger.info(f"Successfully converted '{column}' to integer.")
                        else:
                            df[column] = numeric_col.astype(float)
                            self.logger.info(f"Successfully converted '{column}' to float.")
                    else:
                        df[column] = df[column].astype(str)
                        self.logger.info(f"Converted column '{column}' to string.")
                except Exception as e:
                    self.logger.error(f"Failed to convert column '{column}'. Error: {e}")
        return df

    def _standardize_column_names(self, df):
        """
        Standardizes column names by converting them to lowercase, replacing spaces with underscores, 
        and removing special characters.
        """
        original_columns = df.columns.tolist()
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(' ', '_')  
            .str.replace(r'[^a-z0-9_]', '', regex=False)  
        )
        new_columns = df.columns.tolist()
        self.logger.info(f"Standardized column names from {original_columns} to {new_columns}")
        return df

    def _one_hot_encode(self):
        df_failures = self.dataframes.get('df_failures')
        if df_failures is not None and not df_failures.empty and 'failure' in df_failures.columns:
            df_fail = pd.get_dummies(df_failures, columns=['failure'], dtype=int, prefix='fail').groupby(['datetime', 'machineID']).sum().reset_index()
            df_fail['failure_indicator'] = 1
            self.dataframes['df_failures'] = df_fail
            self.logger.info("One-hot encoding complete for failures.")
        else:
            self.logger.warning("Failures DataFrame is missing or does not contain the 'failure' column.")

        df_errors = self.dataframes.get('df_errors')
        if df_errors is not None and not df_errors.empty and 'errorID' in df_errors.columns:
            df_error = pd.get_dummies(df_errors, columns=['errorID'], dtype=int, prefix='').groupby(['datetime', 'machineID']).sum().reset_index()
            df_error['error_indicator'] = 1
            self.dataframes['df_errors'] = df_error
            self.logger.info("One-hot encoding complete for errors.")
        else:
            self.logger.warning("Errors DataFrame is missing or does not contain the 'errorid' column.")

        df_maint = self.dataframes.get('df_maint')
        if df_maint is not None and not df_maint.empty and 'comp' in df_maint.columns:
            df_main = pd.get_dummies(df_maint, columns=['comp'], dtype=int, prefix='maint').groupby(['datetime', 'machineID']).sum().reset_index()
            df_main['maint_indicator'] = 1
            self.dataframes['df_maint'] = df_main
            self.logger.info("One-hot encoding complete for maintenance.")
        else:
            self.logger.warning("Maintenance DataFrame is missing or does not contain the 'comp' column.")

    def _merge_data(self):
        try:
            df_telemetry = self.dataframes.get('df_telemetry')
            df_failures = self.dataframes.get('df_failures')
            df_errors = self.dataframes.get('df_errors')
            df_maint = self.dataframes.get('df_maint')
            df_machines = self.dataframes.get('df_machines')

            if df_telemetry is None:
                self.logger.error("Telemetry DataFrame is missing.")
                raise ValueError("Telemetry DataFrame is missing.")
            if df_failures is None:
                self.logger.warning("Failures DataFrame is missing. Proceeding without failures.")
            if df_errors is None:
                self.logger.warning("Errors DataFrame is missing. Proceeding without errors.")
            if df_maint is None:
                self.logger.warning("Maintenance DataFrame is missing. Proceeding without maintenance.")
            if df_machines is None:
                self.logger.error("Machines DataFrame is missing.")
                raise ValueError("Machines DataFrame is missing.")

            df_merged = df_telemetry.copy()
            if df_failures is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_failures[['machineID', 'datetime', 'fail_comp1', 'fail_comp2', 'fail_comp3', 'fail_comp4', 'failure_indicator']],
                    on=['machineID', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'failure_indicator': 0, 'fail_comp1': 0, 'fail_comp2': 0, 'fail_comp3': 0, 'fail_comp4': 0}, inplace=True)
                df_merged[['failure_indicator', 'fail_comp1', 'fail_comp2', 'fail_comp3', 'fail_comp4']] = df_merged[
                    ['failure_indicator', 'fail_comp1', 'fail_comp2', 'fail_comp3', 'fail_comp4']
                ].astype(int)

            if df_errors is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_errors[['machineID', 'datetime', '_error1', '_error2', '_error3', '_error4', '_error5', 'error_indicator']],
                    on=['machineID', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'error_indicator': 0, '_error1': 0, '_error2': 0, '_error3': 0, '_error4': 0, '_error5': 0}, inplace=True)

            if df_maint is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_maint[['machineID', 'datetime', 'maint_comp1', 'maint_comp2', 'maint_comp3', 'maint_comp4', 'maint_indicator']],
                    on=['machineID', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'maint_indicator': 0, 'maint_comp1': 0, 'maint_comp2': 0, 'maint_comp3': 0, 'maint_comp4': 0}, inplace=True)

            df_machines = pd.get_dummies(df_machines, columns=['model'], dtype=int, prefix='').groupby(['machineID']).sum().reset_index()
            df_merged = pd.merge(df_merged, df_machines, on=['machineID'], how='left')

            self.logger.info("Data merging complete.")
            return df_merged

        except Exception as e:
            self.logger.error(f"An error occurred during merging: {e}")
            raise


    def _feature_engineering(self, df):
        """
        Performs comprehensive feature engineering including calculations for 
        time_since_last_error, time_since_last_maint, RUL, and NaN handling.

        Args:
            df (DataFrame): The merged DataFrame with telemetry, failure, error, and maintenance info.

        Returns:
            DataFrame: The DataFrame with additional engineered features.
        """
        try:
            # Ensure 'datetime' column is in correct datetime format
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

           
            # Calculate datetime of last error and time since last error
            if 'error_indicator' in df.columns:
                df['datetime_error'] = df['datetime'].where(df['error_indicator'] == 1)
                df['datetime_error'] = df.groupby('machineID')['datetime_error'].bfill()
                df['time_since_last_error'] = (df['datetime_error'] - df['datetime']).dt.total_seconds() / 3600
                

            # Calculate datetime of last maintenance and time since last maintenance
            if 'maint_indicator' in df.columns:
                df['datetime_maint'] = df['datetime'].where(df['maint_indicator'] == 1)
                df['datetime_maint'] = df.groupby('machineID')['datetime_maint'].bfill()
                df['time_since_last_maint'] = (df['datetime_maint'] - df['datetime']).dt.total_seconds() / 3600
               

            # Calculate datetime of next failure and Remaining Useful Life (RUL)
            if 'failure_indicator' in df.columns:
                df['datetime_failure'] = df['datetime'].where(df['failure_indicator'] == 1)
                df['datetime_failure'] = df.groupby('machineID')['datetime_failure'].bfill()
                df['RUL'] = (df['datetime_failure'] - df['datetime']).dt.total_seconds() / 3600
               

            # Fill NaNs in RUL, time_since_last_error, and time_since_last_maint using averages and countdowns
            self._fill_nan_with_countdown(df, 'RUL', 'failure_indicator', 'datetime_failure', 'RUL')
            self._fill_nan_with_countdown(df, 'time_since_last_error', 'error_indicator', 'datetime_error', 'time_since_last_error')
            self._fill_nan_with_countdown(df, 'time_since_last_maint', 'maint_indicator', 'datetime_maint', 'time_since_last_maint')

            # Handle machine-specific RUL adjustments for similar machines (e.g., machine 77 and 6 analogues)
            machine_df = self.dataframes.get('df_machines')  # Retrieve machine_df directly from self.dataframes
            if machine_df is not None:
                self._apply_rul_adjustments(df, machine_df)

            self.logger.info("Feature engineering complete.")
            return df

        except Exception as e:
            self.logger.error(f"An error occurred during feature engineering: {e}")
            raise


    def _fill_nan_with_countdown(self, df, column, indicator, datetime_column, fallback_column):
        """
        Fills NaN values for a specified column by applying average countdowns.

        Args:
            df (DataFrame): The main DataFrame.
            column (str): Column to fill.
            indicator (str): Indicator column (e.g., failure_indicator).
            datetime_column (str): Related datetime column.
            fallback_column (str): Fallback column to compute averages from.
        """
        column_means = df.groupby('machineID')[fallback_column].mean().to_dict()

        for machine_id, mean_value in column_means.items():
            machine_data = df[df['machineID'] == machine_id].copy()
            last_index = machine_data[machine_data[indicator] == 1].index.max()

            if pd.isna(last_index):  # No known events
                initial_value = mean_value
                nan_indices = machine_data.index
            else:  # Events found, start countdown after last event
                initial_value = mean_value
                nan_indices = machine_data.loc[last_index + 1:, column].index

            for idx in nan_indices:
                df.loc[idx, column] = max(initial_value, 0)
                initial_value -= 1
                if initial_value < 0:
                    initial_value = mean_value

    def _apply_rul_adjustments(self, df, machine_df):
        """
        Applies average RUL adjustments based on similar machines for specific cases.

        Args:
            df (DataFrame): The main DataFrame.
            machine_df (DataFrame): The machine information DataFrame.
        """
        # Similar machines for machineID 77 (model 4, age 10-15)
        model4_machines = machine_df[(machine_df['model'] == 'model4') & (machine_df['age'].between(10, 15))]
        model4_rul_mean = df[df['machineID'].isin(model4_machines['machineID'])]['RUL'].dropna().mean()
        self._apply_machine_rul(df, 77, model4_rul_mean)

        # Similar machines for machineID 6 (model 3, age 7)
        model3_machines = machine_df[(machine_df['model'] == 'model3') & (machine_df['age'] == 7)]
        model3_rul_mean = df[df['machineID'].isin(model3_machines['machineID'])]['RUL'].dropna().mean()
        self._apply_machine_rul(df, 6, model3_rul_mean)

    def _apply_machine_rul(self, df, machine_id, mean_rul):
        """
        Applies average RUL countdown for a specific machine.

        Args:
            df (DataFrame): The main DataFrame.
            machine_id (int): Machine ID to adjust RUL.
            mean_rul (float): The mean RUL to use for countdown.
        """
        machine_data = df[df['machineID'] == machine_id].copy()
        initial_value = mean_rul
        nan_indices = machine_data[machine_data['RUL'].isna()].index

        for idx in nan_indices:
            df.loc[idx, 'RUL'] = max(initial_value, 0)
            initial_value -= 1
            if initial_value < 0:
                initial_value = mean_rul


    def _clean_and_split_datetime(self, df):
        """
        Splits 'datetime' column into separate 'date' and 'time' columns.

        Args:
            df (DataFrame): The DataFrame to split the datetime column.

        Returns:
            DataFrame: The DataFrame with 'datetime' split into 'date' and 'time'.
        """
        if 'datetime' in df.columns:
            # Set 'date' as datetime without the time component
            df['date'] = pd.to_datetime(df['datetime']).dt.normalize()
            df['time'] = df['datetime'].dt.strftime('%H:%M')
            df = df.drop(columns=['datetime'])
            self.logger.info("Split 'datetime' column into 'date' and 'time'")
        return df

    def preprocess(self):
        """
        Runs the entire preprocessing pipeline including one-hot encoding, merging, feature engineering, 
        and splitting datetime columns.

        Returns:
            DataFrame: The fully preprocessed DataFrame ready for SQL transfer.
        """
        try:
            
            self._one_hot_encode()
            df_merged = self._merge_data()
            df_engineered = self._feature_engineering(df_merged)
            df_final = self._clean_and_split_datetime(df_engineered)
            self.logger.info("Full preprocessing pipeline complete.")
            return df_final
        
        except Exception as e:
            self.logger.error(f"An error occurred during preprocessing: {e}")
            raise
