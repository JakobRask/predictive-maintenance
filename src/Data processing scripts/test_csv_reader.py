import unittest
import os
import pandas as pd
import sys

# Add the data directory to sys.path to import the CSV module
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data', 'raw')
sys.path.insert(0, data_dir)

from CSV_module import CSVReader

class TestCSVReader(unittest.TestCase):

    def setUp(self):
        # Delete the read log to force re-reading files during the test
        read_files_log = 'files_read_log.txt'
        if os.path.exists(read_files_log):
            os.remove(read_files_log)
            
        self.folder_path = r"C:\Users\Hanss\Documents\Data Science Project\predictive-maintenance\data\raw"
        print(f"Using folder path: {self.folder_path}")
        self.csv_reader = CSVReader(folder_path=self.folder_path)
    def setUp(self):
        """
        Set up the CSVReader instance for testing.
        """
        self.folder_path = r"C:\Users\Hanss\Documents\Data Science Project\predictive-maintenance\data\raw"

        self.csv_reader = CSVReader(folder_path=self.folder_path)

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
            
            # Print out each file name found in the folder
            print(f"Found file: {file_name}")  # <-- This line prints each file found

            if file_name.endswith('.csv') and file_name not in self.read_files:
                try:
                    df = pd.read_csv(file_path)
                    print(f"Successfully read CSV file: {file_name}")  # <-- This line prints each successful read
                    self._store_dataframe(dataframes, file_name, df)
                except Exception as e:
                    self.logger.error(f'Failed to read CSV file {file_name}. Error: {e}')
            elif (file_name.endswith('.xlsx') or file_name.endswith('.xls')) and file_name not in self.read_files:
                try:
                    df = pd.read_excel(file_path)
                    print(f"Successfully read Excel file: {file_name}")  # <-- This line prints each successful read
                    self._store_dataframe(dataframes, file_name, df)
                except Exception as e:
                    self.logger.error(f'Failed to read Excel file {file_name}. Error: {e}')

        return dataframes

    def test_log_file_creation(self):
        """
        Test to ensure that the log file is created and entries are being logged.
        """
        log_file = os.path.join(current_dir, 'csv_reader_log.txt')
        self.assertTrue(os.path.exists(log_file), "Log file does not exist. The logging mechanism may not be working.")

    def test_read_specific_files(self):
        """
        Test that specific files are read into correctly labeled DataFrames.
        """
        dataframes = self.csv_reader.read_files_from_folder()

        # Check if specific keys are present based on the expected file naming convention
        expected_keys = ['df_errors', 'df_failures', 'df_machines', 'df_maintenance', 'df_telemetry']


        for key in expected_keys:
            self.assertIn(key, dataframes, f"Expected key '{key}' not found in dataframes. Check file naming conventions.")

if __name__ == '__main__':
    unittest.main()
