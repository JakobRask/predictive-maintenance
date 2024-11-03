import pandas as pd
import numpy as np


from SQL_module import SQLManager
from PreProcess_module import PreProcessor
from FinalProcessing_module import FinalProcessing
from LSTMModel_Trainer_module import LSTMModelTrainer
from Logging_module import LoggerSetup
from CSV_module import CSVReader  

# Logger setup
logger_setup = LoggerSetup("MainScriptLogger", "main_script_log.log")
logger = logger_setup.get_logger()

FOLDER_PATH = "data/raw"
TIME_STEPS = 24

def main():
    try:
        print("Starting main script")  # Print statement added
        logger.info("Starting main script")

        # Step 1: Data Fetching from CSV Files
        print("Fetching CSV files...")  # Print statement added
        csv_reader = CSVReader(folder_path=FOLDER_PATH)
        dataframes = csv_reader.read_files_from_folder()
        print("CSV files read successfully")  # Print statement added
        logger.info("CSV files read successfully from folder.")

        # Step 2: Data Preprocessing (Complete Pipeline: One-hot Encoding, Merging, Cleaning, etc.)
        print("Starting data preprocessing...")  # Print statement added
        preprocessor = PreProcessor(dataframes)
        df_merged = preprocessor.preprocess()
        print("Data preprocessing complete")  # Print statement added
        logger.info("Data preprocessing complete.")

        # Step 3: Connect to Database and Transfer Data
        print("Connecting to SQL database...")  # Print statement added
        sql_manager = SQLManager('PdM_SQL_manager', 'sql_manager_log.log')
        engine = sql_manager.connect_to_database(dialect='mssql', server='NovaNexus', database='predictive_maintenance_db', integrated_security=True)
        print("Connected to SQL database")  # Print statement added

        # Transfer merged data to SQL
        print("Transferring data to SQL...")  # Print statement added
        sql_manager.transfer_data(df_merged, "merged_data")
        print("Data transfer to SQL complete")  # Print statement added
        logger.info("Data transfer to SQL complete.")

        # Step 4: Fetch Cleaned Data from SQL
        print("Fetching cleaned data from SQL...")  # Print statement added
        df_cleaned = sql_manager.fetch_merged_data("merged_data")
        print("Data fetched from SQL successfully")  # Print statement added
        logger.info("Data fetched from SQL successfully.")

        # Close the database connection
        print("Closing database connection...")  # Print statement added
        sql_manager.close_connection()
        print("Database connection closed")  # Print statement added
        logger.info("Database connection closed.")

        # Step 5: Final Processing - Combine 'date' and 'time' Columns
        print("Combining 'date' and 'time' columns...")  # Print statement added
        df_cleaned['datetime'] = pd.to_datetime(df_cleaned['date'].astype(str) + ' ' + df_cleaned['time'].astype(str))
        df_cleaned = df_cleaned.sort_values(by=['machineID', 'datetime'])
        print("Date and time columns merged into 'datetime'")  # Print statement added
        logger.info("Date and time columns merged into 'datetime'.")

        # Step 6: Scaling and Sequence Generation
        print("Starting data scaling and sequence generation...")  # Print statement added
        final_processor = FinalProcessing()  # Initialize FinalProcessing
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = final_processor.split_scale_sequence(df_cleaned)
        print("Data scaling and sequence generation complete")  # Print statement added
        logger.info("Data scaling and sequence generation complete.")

        print("Starting model training...")  # Print statement added
        model_trainer = LSTMModelTrainer(time_steps=TIME_STEPS, features=X_train_seq.shape[2])
        model_trainer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        print("Model training complete")  # Print statement added
        logger.info("Model training complete.")

        # Step 8: Model Evaluation and Saving
        print("Evaluating model...")  # Print statement added
        test_loss, test_mae = model_trainer.evaluate(X_test_seq, y_test_seq)
        print(f"Model evaluation complete. Test Loss: {test_loss}, MAE: {test_mae}")  # Print statement added
        logger.info(f"Model evaluation complete. Test Loss: {test_loss}, MAE: {test_mae}")
        
        

    except Exception as e:
        print(f"An error occurred: {e}")  # Print statement added
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
