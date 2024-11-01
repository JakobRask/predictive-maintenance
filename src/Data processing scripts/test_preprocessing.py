import unittest
import pandas as pd
from PreProcess_module import PreProcessor
from CSV_module import CSVReader  
class TestPreProcessor(unittest.TestCase):

    def setUp(self):
        """
        Set up test data by reading in required CSV files using CSVReader.
        """
        
        folder_path = r"C:\Users\Hanss\Documents\Data Science Project\predictive-maintenance\data\raw"
        
        
        csv_reader = CSVReader(folder_path=folder_path)
        self.dataframes = csv_reader.read_files_from_folder()

        
        # Initialize PreProcessor with the real dataframes
        self.preprocessor = PreProcessor(self.dataframes)

    def test_remove_duplicates(self):
        """
        Test that duplicates are removed from real DataFrame.
        """
        for df_key, df in self.dataframes.items():
            df_no_duplicates = self.preprocessor._remove_duplicates(df)
            self.assertLessEqual(len(df_no_duplicates), len(df), f"Duplicates were not removed correctly from {df_key}.")
            self.assertEqual(df_no_duplicates.duplicated().sum(), 0, f"Duplicates still exist in {df_key} after removal.")

    def test_change_column_dtypes(self):
        """
        Test that column data types are changed correctly in real data.
        """
        for df_key, df in self.dataframes.items():
            df_changed = self.preprocessor._change_column_dtypes(df)
            if 'datetime' in df_changed.columns:
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_changed['datetime']), f"Datetime column was not converted correctly in {df_key}.")
            # Add further checks for integer and float conversion

    def test_standardize_column_names(self):
        """
        Test that column names are standardized correctly in real data.
        """
        for df_key, df in self.dataframes.items():
            df_standardized = self.preprocessor._standardize_column_names(df)
            self.assertTrue(all([col == col.lower().replace(' ', '_') for col in df_standardized.columns]), f"Column names were not standardized correctly in {df_key}.")
            self.assertTrue(all([not any(char in col for char in ['-', ' ']) for col in df_standardized.columns]), f"Special characters were not removed in {df_key}.")

    def test_one_hot_encode(self):
        """
        Test that one-hot encoding is performed correctly on real data.
        """
        # Run the one-hot encoding method
        self.preprocessor._one_hot_encode()

        # Check for the presence of the expected DataFrames
        self.assertIn('df_failures', self.preprocessor.dataframes, "Failures DataFrame was not one-hot encoded correctly.")
        self.assertIn('df_errors', self.preprocessor.dataframes, "Errors DataFrame was not one-hot encoded correctly.")
        self.assertIn('df_maintenance', self.preprocessor.dataframes, "Maintenance DataFrame was not one-hot encoded correctly.")
        
        # Ensure these DataFrames are not None
        self.assertIsNotNone(self.preprocessor.dataframes.get('df_failures'), "Failures DataFrame is None after one-hot encoding.")
        self.assertIsNotNone(self.preprocessor.dataframes.get('df_errors'), "Errors DataFrame is None after one-hot encoding.")
        self.assertIsNotNone(self.preprocessor.dataframes.get('df_maintenance'), "Maintenance DataFrame is None after one-hot encoding.")


    def test_merge_data(self):
        """
        Test that merging is performed correctly with real data.
        """
        self.preprocessor._one_hot_encode()  # Make sure to encode first
        df_merged = self.preprocessor._merge_data()
        self.assertIsNotNone(df_merged, "Merged DataFrame is None.")
        self.assertIn('machineid', df_merged.columns, "'machineid' not found in merged DataFrame.")
        self.assertIn('datetime', df_merged.columns, "'datetime' not found in merged DataFrame.")



    def test_feature_engineering(self):
        """
        Test that feature engineering is performed correctly on real data.
        """
        # One-hot encode and merge before feature engineering
        self.preprocessor._one_hot_encode()
        df_merged = self.preprocessor._merge_data()

        # Perform feature engineering
        df_features = self.preprocessor._feature_engineering(df_merged)
        self.assertIn('time_since_last_error', df_features.columns, "'time_since_last_error' not found in engineered DataFrame.")
        self.assertIn('RUL', df_features.columns, "'RUL' not found in engineered DataFrame.")

    

    def test_full_pipeline(self):
        """
        Test that the entire preprocessing pipeline works without errors using real data.
        """
        df_preprocessed = self.preprocessor.preprocess()
        self.assertFalse(df_preprocessed.empty, "The final preprocessed DataFrame is empty.")
        self.assertIn('machineid', df_preprocessed.columns, "'machineid' column not found in final preprocessed DataFrame.")
        self.assertIn('date', df_preprocessed.columns, "'date' column not found in final preprocessed DataFrame.")

if __name__ == '__main__':
    unittest.main()
