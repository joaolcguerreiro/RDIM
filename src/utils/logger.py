import sys
import logging


class TeeLogger(object):
    def __init__(self, log_file_path, original_stdout):
        self.log_file_path = log_file_path
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        with open(self.log_file_path, 'a') as f:
            f.write(message)

    def flush(self):
        self.original_stdout.flush()

    def close(self):
        self.original_stdout.close()

    def __del__(self):
        self.close()


def setup_logging(log_file_path, mode='w'):
    # Clear file content based on access mode
    with open(log_file_path, mode):
        pass

    #=======================================#
    #   Redirect logging messages to file   #
    #=======================================#

    # Create a logger object
    logger = logging.getLogger()

    # Remove all handlers associated with the root logger object (console/stream handler)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a handler for writing to the log file
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    #=============================================#
    #   Tee stdout messages to console and file   #
    #=============================================#

    # Replace stdout with custom teeing logger
    sys.stdout = TeeLogger(log_file_path, sys.stdout)


def print_divisor(width=140, last=False):
    print('=' * width + ('\n\n' if last else ''))
