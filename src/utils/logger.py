import logging
import sys


class PrintLogger(object):
    def __init__(self, logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        if message != '\n':
            self.logger.info(message)

    def flush(self):
        self.original_stdout.flush()


def setup_logger(log_file_path):
    # Create a logger object
    logger = logging.getLogger()

    # Remove all handlers associated with the root logger object.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a handler for writing to the log file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    # Replace sys.stdout with our custom PrintLogger
    sys.stdout = PrintLogger(logger, sys.stdout)
