import logging
from datetime import datetime

class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.logger = logging.getLogger("singleton_logger")
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def get_logger(self):
        return self.logger

class CustomFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = record.getMessage()
        return f"[{level}] {time} - {message}"

logger = SingletonLogger().get_logger()