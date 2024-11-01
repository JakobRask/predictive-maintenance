import os
import pandas as pd
from Logging_module import LoggerSetup  

class CSVReader:
    """
    The 'CSVReader' reads new Excel or CSV files from a specified folder and saves the information into separate DataFrames.
    """
    
    def __init__(self, folder_path,
                 read_files_log='files_read_log.txt', 
                 log_file='csv_reader_log.txt'):
        """
        Initializes CSVReader with local folder path and log files.

        Args:
            folder_path (str): Path to folder containing local CSV or Excel files.
            read_files_log (str): File to log already read files.
            log_file (str): Log file for logging events.
        """
        self.folder_path = folder_path
        self.read_files_log = read_files_log
        
        logger_name = 'CSVReaderLogger'  
        logger_setup = LoggerSetup(logger_name, log_file)
        self.logger = logger_setup.get_logger()
        
        self.read_files = self._get_read_files()

    def _get_read_files(self):
        """
        Returns a set of already read file names from the log.

        Returns:
            set: Set of read file names.
        """
        if os.path.exists(self.read_files_log):
            with open(self.read_files_log, 'r') as f:
                self.logger.info(f'Reading processed files from log: {self.read_files_log}')
                return set(f.read().splitlines())
        else:
            self.logger.info('No log file found.')
            return set()

    def _update_read_files_log(self, file_name):
        """
        Adds a new file name to the log.

        Args:
            file_name (str): Name of the file to add.
        """
        with open(self.read_files_log, 'a') as f:
            f.write(f'{file_name}\n')
        self.logger.info(f'Updated log with file: {file_name}')

    def read_files_from_folder(self):
        """
        Reads new Excel or CSV files from the local folder.

        Returns:
            dict: A dictionary with the names of the files as keys and their tables as values.
        """
        dataframes = {}

        if not self.folder_path:
            self.logger.warning('No folder path specified for reading local files.')
            return dataframes

        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if file_name.endswith('.csv') and file_name not in self.read_files:
                try:
                    df = pd.read_csv(file_path)
                    self._store_dataframe(dataframes, file_name, df)
                except Exception as e:
                    self.logger.error(f'Failed to read CSV file {file_name}. Error: {e}')
            elif (file_name.endswith('.xlsx') or file_name.endswith('.xls')) and file_name not in self.read_files:
                try:
                    df = pd.read_excel(file_path)
                    self._store_dataframe(dataframes, file_name, df)
                except Exception as e:
                    self.logger.error(f'Failed to read Excel file {file_name}. Error: {e}')

        return dataframes
    
    def _store_dataframe(self, dataframes, file_name, df):
        """
        Helper function to store dataframes with descriptive keys.

        Args:
            dataframes (dict): Dictionary to store the dataframes.
            file_name (str): The name of the file.
            df (DataFrame): The DataFrame read from the file.
        """
        base_name = file_name.replace('.csv', '').replace('.xlsx', '').replace('.xls', '').lower()

        if 'errors' in base_name:
            dataframes['df_errors'] = df
        elif 'failures' in base_name:
            dataframes['df_failures'] = df
        elif 'machines' in base_name:
            dataframes['df_machines'] = df
        elif 'maint' in base_name:
            dataframes['df_maintenance'] = df
        elif 'telemetry' in base_name:
            dataframes['df_telemetry'] = df
        else:
            
            cleaned_name = base_name.replace('pdm_', '')
            dataframes[cleaned_name] = df
            self.logger.warning(f'File {file_name} did not match any predefined categories and is stored as {cleaned_name}.')

        self.logger.info(f'Successfully read file from folder: {file_name}')
        self._update_read_files_log(file_name)



