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
        """
        Performs one-hot encoding for failures, errors, and maintenance on the respective DataFrames.
        """
        # One-hot encode failures
        df_failures = self.dataframes.get('df_failures')
        if df_failures is not None and not df_failures.empty and 'failure' in df_failures.columns:
            df_fail = pd.get_dummies(df_failures, columns=['failure'], dtype=int, prefix='fail').groupby(['datetime', 'machineid']).sum().reset_index()
            df_fail['failure_indicator'] = 1
            self.dataframes['df_failures'] = df_fail
            self.logger.info("One-hot encoding complete for failures.")
        else:
            self.logger.warning("Failures DataFrame is missing or does not contain the 'failure' column.")

        # One-hot encode errors
        df_errors = self.dataframes.get('df_errors')
        if df_errors is not None and not df_errors.empty and 'errorid' in df_errors.columns:
            df_error = pd.get_dummies(df_errors, columns=['errorid'], dtype=int, prefix='').groupby(['datetime', 'machineid']).sum().reset_index()
            df_error['error_indicator'] = 1
            self.dataframes['df_errors'] = df_error
            self.logger.info("One-hot encoding complete for errors.")
        else:
            self.logger.warning("Errors DataFrame is missing or does not contain the 'errorid' column.")

        # One-hot encode maintenance
        df_maintenance = self.dataframes.get('df_maintenance')
        if df_maintenance is not None and not df_maintenance.empty and 'comp' in df_maintenance.columns:
            df_main = pd.get_dummies(df_maintenance, columns=['comp'], dtype=int, prefix='maint').groupby(['datetime', 'machineid']).sum().reset_index()
            df_main['maint_indicator'] = 1
            self.dataframes['df_maintenance'] = df_main
            self.logger.info("One-hot encoding complete for maintenance.")
        else:
            self.logger.warning("Maintenance DataFrame is missing or does not contain the 'comp' column.")



    def _merge_data(self):
        """
        Merges all relevant DataFrames into a single DataFrame.

        Returns:
            DataFrame: The merged DataFrame.
        """
        try:
            df_telemetry = self.dataframes.get('df_telemetry')
            df_failures = self.dataframes.get('df_failures')
            df_errors = self.dataframes.get('df_errors')
            df_maintenance = self.dataframes.get('df_maintenance')
            df_machines = self.dataframes.get('df_machines')

            # Check for NoneType DataFrames
            if df_telemetry is None:
                self.logger.error("Telemetry DataFrame is missing.")
                raise ValueError("Telemetry DataFrame is missing.")
            if df_failures is None:
                self.logger.warning("Failures DataFrame is missing. Proceeding without failures.")
            if df_errors is None:
                self.logger.warning("Errors DataFrame is missing. Proceeding without errors.")
            if df_maintenance is None:
                self.logger.warning("Maintenance DataFrame is missing. Proceeding without maintenance.")
            if df_machines is None:
                self.logger.error("Machines DataFrame is missing.")
                raise ValueError("Machines DataFrame is missing.")

            # Merge telemetry with failure data
            df_merged = df_telemetry.copy()
            if df_failures is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_failures[['machineid', 'datetime', 'fail_comp1', 'fail_comp2', 'fail_comp3', 'fail_comp4', 'failure_indicator']],
                    on=['machineid', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'failure_indicator': 0, 'fail_comp1': 0, 'fail_comp2': 0, 'fail_comp3': 0, 'fail_comp4': 0}, inplace=True)

            # Merge with error data
            if df_errors is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_errors[['machineid', 'datetime', '_error1', '_error2', '_error3', '_error4', '_error5', 'error_indicator']],
                    on=['machineid', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'error_indicator': 0, '_error1': 0, '_error2': 0, '_error3': 0, '_error4': 0, '_error5': 0}, inplace=True)

            # Merge with maintenance data (if available)
            if df_maintenance is not None:
                df_merged = pd.merge(
                    df_merged,
                    df_maintenance[['machineid', 'datetime', 'maint_comp1', 'maint_comp2', 'maint_comp3', 'maint_comp4', 'maint_indicator']],
                    on=['machineid', 'datetime'],
                    how='left'
                )
                df_merged.fillna({'maint_indicator': 0, 'maint_comp1': 0, 'maint_comp2': 0, 'maint_comp3': 0, 'maint_comp4': 0}, inplace=True)

            # Add machine details
            df_machines = pd.get_dummies(df_machines, columns=['model'], dtype=int, prefix='').groupby(['machineid']).sum().reset_index()
            df_merged = pd.merge(df_merged, df_machines, on=['machineid'], how='left')

            self.logger.info("Data merging complete.")
            return df_merged

        except Exception as e:
            self.logger.error(f"An error occurred during merging: {e}")
            raise





    def _feature_engineering(self, df):
        """
        Performs feature engineering including calculations for time_since_last_error, time_since_last_maint, and RUL.

        Args:
            df (DataFrame): The merged DataFrame.

        Returns:
            DataFrame: The DataFrame with additional engineered features.
        """
        try:
            # Calculate time since last error
            df['datetime_error'] = df['datetime'].where(df['error_indicator'] == 1)
            df['datetime_error'] = df.groupby('machineid')['datetime_error'].bfill()
            df['time_since_last_error'] = (df['datetime_error'] - df['datetime']).dt.total_seconds() / 3600

            # Calculate time since last maintenance
            df['datetime_maint'] = df['datetime'].where(df['maint_indicator'] == 1)
            df['datetime_maint'] = df.groupby('machineid')['datetime_maint'].bfill()
            df['time_since_last_maint'] = (df['datetime_maint'] - df['datetime']).dt.total_seconds() / 3600

            # Calculate Remaining Useful Life (RUL)
            df['datetime_failure'] = df['datetime'].where(df['failure_indicator'] == 1)
            df['datetime_failure'] = df.groupby('machineid')['datetime_failure'].bfill()
            df['RUL'] = (df['datetime_failure'] - df['datetime']).dt.total_seconds() / 3600

            self.logger.info("Feature engineering complete.")
            return df

        except Exception as e:
            self.logger.error(f"An error occurred during feature engineering: {e}")
            raise

    def _clean_and_split_datetime(self, df):
        """
        Splits 'datetime' column into separate 'date' and 'time' columns.

        Args:
            df (DataFrame): The DataFrame to split the datetime column.

        Returns:
            DataFrame: The DataFrame with 'datetime' split into 'date' and 'time'.
        """
        if 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.strftime('%H:%M')
            df = df.drop(columns=['datetime'])
            self.logger.info("Split 'datetime' column into 'date' and 'time")
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
