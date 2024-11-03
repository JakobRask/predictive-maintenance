import logging



class LoggerSetup:
    def __init__(self, logger_name, log_file):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.hasHandlers():  
            self.setup_logger(log_file)

    def setup_logger(self, log_file):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

    def get_logger(self):
        # Ensure the logger is properly configured and has at least one handler.
        if not self.logger.handlers:
            raise ValueError("Logger has no handlers configured.")
        return self.logger
