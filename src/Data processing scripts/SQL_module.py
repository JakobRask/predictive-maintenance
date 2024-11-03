import sys
sys.path.append('src/Data processing scripts')

from sqlalchemy import create_engine, inspect
import pandas as pd
from Logging_module import LoggerSetup

"""
SQLManager Module

This module helps you work with databases. It has a class called SQLManager that
lets you:
1. Connect to a database.
2. Save merged data to the database.
3. Fetch merged data from the database.

Steps to use:
1. Make an SQLManager object.
2. Connect to your database using connect_to_database().
3. Use transfer_data() to save merged data to the database.
4. Use fetch_merged_data() to get data from the database.

External libraries: 
- sqlalchemy: Used to connect to and interact with SQL databases.
- pandas: Used to handle and process data.
- logging_module (LoggerSetup): For logging API interactions and errors.
"""

class SQLManager:
    """
    SQLManager class handles connection to a SQL database, fetching merged data,
    and saving merged data to the database.
    """

    def __init__(self, logger_name, log_file):
        """
        Initializes the SQLManager class by setting up the logger.

        Parameters:
        logger_name (str): The name of the logger.
        log_file (str): The file where logs are written.
        """
        logger_setup = LoggerSetup(logger_name, log_file)
        self.logger = logger_setup.get_logger()
        self.engine = None

    def connect_to_database(self, dialect, server, database, user=None,
                            password=None, integrated_security=True):
        """
        Creates a SQLAlchemy engine for a database connection.

        Parameters:
        dialect (str): The type of database (e.g., 'mssql' for Microsoft SQL Server).
        server (str): The server or computer where the database is hosted.
        database (str): The name of the database.
        user (str, optional): The username for authentication 
                              (if not using Windows login).
        password (str, optional): The password for authentication 
                                  (if not using Windows login).
        integrated_security (bool): Set to True for Windows authentication; 
                                    False for username/password authentication.

        Returns:
        A SQLAlchemy engine connected to the specified database.
        """
        try:
            if integrated_security:
                # For Windows authentication
                eng = (f'{dialect}://{server}/{database}'
                       '?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
            else:
                # For SQL Server authentication
                eng = (f'{dialect}://{user}:{password}@{server}/{database}'
                       '?driver=ODBC+Driver+17+for+SQL+Server')

            self.logger.info(f'{dialect} Engine created with connection string: {eng}')
            self.engine = create_engine(eng)
            self.logger.info('Connection to SQL Server successful.')
            return self.engine

        except Exception as e:
            self.logger.error(f'Failed to create engine: {e}')
            raise

    def transfer_data(self, df, table_name, dtype=None):
        """
        Transfers a pandas DataFrame to the connected database table.

        Parameters:
        df (Pandas DataFrame): The DataFrame containing merged data to save.
        table_name (str): The name of the table where the data will be saved.

        If the table already exists, the data will be appended. Otherwise, 
        a new table will be created.
        """
        if self.engine is None:
            self.logger.error('No engine created. Call connect_to_database() first.')
            raise Exception('No engine created. Call connect_to_database() first.')

        try:
            inspector = inspect(self.engine)
            if table_name in inspector.get_table_names():
                self.logger.info(f'Table "{table_name}" exists. Appending merged data.')
                df.to_sql(table_name, con=self.engine, if_exists='append', index=False, dtype=dtype)
            else:
                self.logger.info(f'Table "{table_name}" does not exist. Creating new table with merged data.')
                df.to_sql(table_name, con=self.engine, if_exists='replace', index=False, dtype=dtype)
        except Exception as e:
            self.logger.error(f'Error transferring merged data to table named "{table_name}": {e}')
            raise

    def fetch_merged_data(self, table_name):
        """
        Fetches the merged data from the connected database using the table name.

        Parameters:
        table_name (str): The name of the table to fetch data from.

        Returns:
        DataFrame: The resulting merged data from the table in a pandas DataFrame.
        """
        if self.engine is None:
            self.logger.error('No engine created. Call connect_to_database() first.')
            raise Exception('No engine created. Call connect_to_database() first.')

        try:
            self.logger.info(f'Fetching data from table: {table_name}')
            with self.engine.connect() as connection:
                df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
            self.logger.info(f'Data fetched from table "{table_name}" successfully.')
            return df
        except Exception as e:
            self.logger.error(f'There was an error in fetching data from table "{table_name}": {e}')
            raise

    def close_connection(self):
        """
        Dispose of the SQLAlchemy engine to close the connection to the database.
        """
        if self.engine:
            self.engine.dispose()
            self.logger.info('Database connection closed.')
