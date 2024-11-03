import unittest
import os
import pandas as pd
import sys

# Add the main project directory to sys.path to import the CSV module if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.insert(0, project_dir)

from CSV_module import CSVReader

class TestCSVReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define shared attributes for the class
        cls.folder_path = r"C:\Users\Hanss\Documents\Data Science Project\predictive-maintenance\data\raw"
        cls.read_files_log = 'files_read_log.txt'
        cls.log_file = 'csv_reader_log.txt'

        # Remove read_files_log if it exists to ensure a fresh test
        if os.path.exists(cls.read_files_log):
            os.remove(cls.read_files_log)

        # Create a CSVReader instance
        cls.csv_reader = CSVReader(folder_path=cls.folder_path)

    def test_log_file_creation(self):
        """
        Test to ensure that the log file is created and entries are being logged.
        """
        # Perform some actions that should trigger logging
        self.csv_reader.read_files_from_folder()

        # Check if the log file exists after the test
        self.assertTrue(os.path.exists(self.log_file), "Log file does not exist. The logging mechanism may not be working.")

    def test_read_specific_files(self):
        """
        Test that specific files are read into correctly labeled DataFrames.
        """
        dataframes = self.csv_reader.read_files_from_folder()

        # Check if specific keys are present based on the expected file naming convention
        expected_keys = ['df_errors', 'df_failures', 'df_machines', 'df_maint', 'df_telemetry']
        for key in expected_keys:
            self.assertIn(key, dataframes, f"Expected key '{key}' not found in dataframes. Check file naming conventions.")

if __name__ == '__main__':
    unittest.main()
