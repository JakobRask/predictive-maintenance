import pandas as pd
from Logging_module import LoggerSetup


class PreProcessor:
    """
    A class to perform data preprocessing and cleaning on DataFrames.
    """
    
    def __init__(self, df, logger_name='data_preprocesser_logger', log_file='data_preprocessor.log'):
        """
        Initializes the DataCleaner with a DataFrame and sets up a log to track data transformations.

        Args:
            df: The DataFrame to be cleaned.
            logger_name (str): The name of the logger.
            log_file (str): The file where the log output will be saved.
        """
        
        self.df = df
        logger_setup = LoggerSetup(logger_name=logger_name, log_file=log_file)
        self.logger = logger_setup.get_logger()

       

    def _remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.
        """
        original_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        final_rows = len(self.df)
        duplicates_removed = original_rows - final_rows
        self.logger.info(f"Removed {duplicates_removed} duplicates. Final number of rows: {final_rows}")

    

    def _change_column_dtypes(self):
        """
        Changes data types of the columns based on their content:
        
        - Integer-like columns are converted to int.
        - Decimal-like columns are converted to float.
        - Datetime seperated to date and time.
        - All other columns are converted to str.
        """
        for column in self.df.columns:
            if column == 'datetime':
                try:
                    self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')

                    if pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
                        
                        self.df['date'] = self.df['datetime'].dt.date
                        self.df['time'] = self.df['datetime'].dt.strftime('%H:%M')
                        self.df.drop(columns=['datetime'], inplace=True)
                        
                        self.logger.info(f"Split 'datetime' column into 'date' and 'time'")
                    else:
                        self.logger.warning(f"Column 'datetime' could not be converted to datetime.")

                except Exception as e:
                    self.logger.error(f"Failed to convert column 'datetime' to datetime data type. Error: {e}")

            else:
                try:
                    numeric_col = pd.to_numeric(self.df[column], errors='coerce')

                    if numeric_col.notna().all():
                        
                        if (numeric_col % 1 == 0).all():
                            self.df[column] = numeric_col.astype(int)
                            self.logger.info(f"Successfully converted '{column}' to integer.")
                        else:
                            self.df[column] = numeric_col.astype(float)
                            self.logger.info(f"Successfully converted '{column}' to float.")
                    else:
                        self.df[column] = self.df[column].astype(str)
                        self.logger.info(f"Converted column '{column}' to string.")
                
                except Exception as e:
                    self.logger.error(f"Failed to convert column '{column}'. Error: {e}")


    def _standardize_column_names(self):
        """
        Standardizes column names by converting them to lowercase, replacing spaces with underscores, 
        and removing special characters.
        """
        original_columns = self.df.columns.tolist()
        self.df.columns = (
                self.df.columns
                .str.replace(' ', '_')  
                .str.replace(r'[^a-z0-9_]', '', regex=False)  
        )
        
        new_columns = self.df.columns.tolist()
        self.logger.info(f"Standardized column names from {original_columns} to {new_columns}")

    def _reorder_columns(self, new_order):
        """
        Reorders the columns in the DataFrame according to the specified order.

        Parameters:
        new_order (list): List of column names in the order desired.

        Returns:
        The DataFrame with columns reordered.
        """
        try:
            self.df = self.df.reindex(columns=new_order)
            self.logger.info(f"Columns reordered to: {new_order}")
        except KeyError as e:
            self.logger.error(f"Error: Could not reorder columns. {e}")
        
        return self.df
    
    def clean_data(self):
        """
        Apply the complete data cleaning process to the DataFrame.

        Returns:
            DataFRame: The cleaned DataFrame.
        """
        column_order= ['machineID', 'date', 'time', 'errorID']
        self._change_column_dtypes()
        self._standardize_column_names()
        self._remove_duplicates()
              
        
        return self.df