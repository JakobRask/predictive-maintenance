import sys
sys.path.append('src/Data processing scripts')

from SQL_module import SQLManager  
from CSV_module import CSVReader  
from PreProcess_module import PreProcessor
from Logging_module import LoggerSetup  
from sqlalchemy.types import SmallInteger, String, Date, Time



file_urls = [
    "https://raw.githubusercontent.com/JakobRask/predictive-maintenance/main/data/raw/PdM_errors.csv",
    "https://raw.githubusercontent.com/JakobRask/predictive-maintenance/main/data/raw/PdM_failures.csv",
    "https://raw.githubusercontent.com/JakobRask/predictive-maintenance/main/data/raw/PdM_machines.csv",
    "https://raw.githubusercontent.com/JakobRask/predictive-maintenance/main/data/raw/PdM_maint.csv",
    "https://raw.githubusercontent.com/JakobRask/predictive-maintenance/main/data/raw/PdM_telemetry.csv"
]


sql_manager = SQLManager('SQLManagerLogger', 'sql_manager_log.log')


engine = sql_manager.new_engine(
    dialect='mssql',  
    server='NovaNexus',  
    database='predictive_maintenance_db',  
    integrated_security=True  
)


csv_reader = CSVReader(file_urls, 'csv_reader_log.log')
dataframes = csv_reader.read_csv_files()


for key, df in dataframes.items():
    
    preprocessor = PreProcessor(df)
    cleaned_df = preprocessor.clean_data()

   
    sql_manager.transfer_data(cleaned_df, table_name=key, dtype={
    'machineID': SmallInteger(),
    'errorID': String(),
    'date': Date(),
    'time': Time()  
    })


    print(f"Successfully transferred the data from '{key}' to the SQL database.")
