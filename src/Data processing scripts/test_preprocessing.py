import unittest
import os
import pandas as pd
from PreProcess_module import PreProcessor
from CSV_module import CSVReader

class TestPreProcessor(unittest.TestCase):

    def setUp(self):
        """
        Set up test data by reading in required CSV or Excel files using CSVReader.
        """
        folder_path = r"C:\Users\Hanss\Documents\Data Science Project\predictive-maintenance\data\raw"
        read_files_log = 'files_read_log.txt'
        if os.path.exists(read_files_log):
            os.remove(read_files_log)
        
        csv_reader = CSVReader(folder_path=folder_path, read_files_log=read_files_log)
        self.dataframes = csv_reader.read_files_from_folder()
        self.preprocessor = PreProcessor(self.dataframes)

    def test_remove_duplicates(self):
        """
        Test that duplicates are removed from each DataFrame in the data.
        """
        for df_key, df in self.dataframes.items():
            df_no_duplicates = self.preprocessor._remove_duplicates(df)
            self.assertLessEqual(len(df_no_duplicates), len(df), f"Duplicates were not removed correctly from {df_key}.")
            self.assertEqual(df_no_duplicates.duplicated().sum(), 0, f"Duplicates still exist in {df_key} after removal.")

    def test_change_column_dtypes(self):
        """
        Test that column data types are changed correctly based on column content.
        """
        for df_key, df in self.dataframes.items():
            df_changed = self.preprocessor._change_column_dtypes(df)
            if 'datetime' in df_changed.columns:
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_changed['datetime']), f"Datetime column was not converted correctly in {df_key}.")
            
            for column in df_changed.columns:
                if df_changed[column].dtype == int:
                    self.assertTrue(pd.api.types.is_integer_dtype(df_changed[column]), f"Column '{column}' was not converted to integer correctly in {df_key}.")
                elif df_changed[column].dtype == float:
                    self.assertTrue(pd.api.types.is_float_dtype(df_changed[column]), f"Column '{column}' was not converted to float correctly in {df_key}.")

    def test_standardize_column_names(self):
        """
        Test that column names are standardized correctly in each DataFrame.
        """
        for df_key, df in self.dataframes.items():
            df_standardized = self.preprocessor._standardize_column_names(df)
            self.assertTrue(all([col == col.lower().replace(' ', '_') for col in df_standardized.columns]), f"Column names were not standardized correctly in {df_key}.")
            self.assertTrue(all([not any(char in col for char in ['-', ' ']) for col in df_standardized.columns]), f"Special characters were not removed in {df_key}.")
            self.assertTrue(all(col.isidentifier() for col in df_standardized.columns), f"Invalid characters remain in columns of {df_key}.")

    def test_one_hot_encode(self):
        """
        Test that one-hot encoding is performed correctly for failures, errors, and maintenance.
        """
        self.preprocessor._one_hot_encode()

        df_failures = self.preprocessor.dataframes.get('df_failures')
        df_errors = self.preprocessor.dataframes.get('df_errors')
        df_maint = self.preprocessor.dataframes.get('df_maint')

        self.assertIsNotNone(df_failures, "Failures DataFrame is None after one-hot encoding.")
        self.assertIsNotNone(df_errors, "Errors DataFrame is None after one-hot encoding.")
        self.assertIsNotNone(df_maint, "Maintenance DataFrame is None after one-hot encoding.")
        
        self.assertTrue(any(col.startswith('fail_') for col in df_failures.columns), "One-hot encoding columns for failures are missing.")
        self.assertTrue(any(col.startswith('_error') for col in df_errors.columns), "One-hot encoding columns for errors are missing.")
        self.assertTrue(any(col.startswith('maint_') for col in df_maint.columns), "One-hot encoding columns for maintenance are missing.")

    def test_merge_data(self):
        """
        Test that merging is performed correctly with real data.
        """
        self.preprocessor._one_hot_encode()  
        df_merged = self.preprocessor._merge_data()
        self.assertIsNotNone(df_merged, "Merged DataFrame is None.")
        self.assertIn('machineID', df_merged.columns, "'machineID' not found in merged DataFrame.")
        self.assertIn('datetime', df_merged.columns, "'datetime' not found in merged DataFrame.")

        for col in ['failure_indicator', 'error_indicator', 'maint_indicator']:
            self.assertIn(col, df_merged.columns, f"'{col}' column not found in merged DataFrame.")
            self.assertTrue((df_merged[col] >= 0).all(), f"'{col}' column contains invalid values in merged DataFrame.")

    def test_feature_engineering(self):
        """
        Test that feature engineering is performed correctly, including date range filtering and RUL calculation.
        """
        self.preprocessor._one_hot_encode()
        df_merged = self.preprocessor._merge_data()
        df_features = self.preprocessor._feature_engineering(df_merged)

        self.assertIn('time_since_last_error', df_features.columns, "'time_since_last_error' not found in engineered DataFrame.")
        self.assertIn('RUL', df_features.columns, "'RUL' not found in engineered DataFrame.")

        self.assertTrue(df_features['time_since_last_error'].notna().all(), "'time_since_last_error' contains NaN values.")
        self.assertTrue(df_features['RUL'].notna().all(), "'RUL' contains NaN values.")
        
       

    def test_full_pipeline(self):
        """
        Test that the entire preprocessing pipeline works without errors using real data.
        """
        df_preprocessed = self.preprocessor.preprocess()
        self.assertFalse(df_preprocessed.empty, "The final preprocessed DataFrame is empty.")
        self.assertIn('machineID', df_preprocessed.columns, "'machineID' column not found in final preprocessed DataFrame.")
        self.assertIn('date', df_preprocessed.columns, "'date' column not found in final preprocessed DataFrame.")
        self.assertIn('time', df_preprocessed.columns, "'time' column not found in final preprocessed DataFrame.")

        # Check for correct formats of 'date' and 'time' columns
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_preprocessed['date']), "'date' column is not in datetime format.")
        self.assertTrue(all(df_preprocessed['time'].str.match(r'^\d{2}:\d{2}$')), "'time' column format is incorrect.")

if __name__ == '__main__':
    unittest.main()
