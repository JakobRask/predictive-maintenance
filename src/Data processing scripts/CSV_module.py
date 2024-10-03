import pandas as pd
import requests
import io
from Logging_module import LoggerSetup

class CSVReader:
    """
    The 'CSVReader' reads notebooks (CSV files) from the internet (GitHub) and saves
    the information from each into a table (DataFrame).
    """
    
    def __init__(self, file_urls, log_file):
        """
        Sets up the CSVReader with a list of URLs and the log file to write down what happens.

        Args:
            file_urls (list): A list of internet links to the CSV files (notebooks).
            log_file (str): The name of the file where the log is kept.
        """
        
        self.file_urls = file_urls
        logger_setup = LoggerSetup("CSVReaderLogger", log_file)
        self.logger = logger_setup.get_logger()

    def read_csv_files(self):
        """
        Opens each CSV file, reads the information, and saves it into a DataFrame.
        
        Returns:
            dict: A dictionary with the names of the csv files as keys and their tables as values.
        """
        
        dataframes = {}

       
        for url in self.file_urls:
            file_name = url.split('/')[-1].replace('.csv', '')
            
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))

                dataframes[file_name] = df
                self.logger.info(f'Successfully read file: {file_name}')
            else:
                self.logger.error(f"Failed to fetch {file_name}. HTTP Status Code: {response.status_code}")
        
        return dataframes






