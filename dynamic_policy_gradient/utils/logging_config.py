import logging
import os

def setup_logging():
    # create the logs function if not exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Check if a console handler is already added and only add if there isn't one
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        logger.addHandler(console_handler)

    # Initially add a file handler
    # add_file_handler(logger, "logs/softmax_pg.log")


def add_file_handler(logger, filename):
    # Create a file handler that logs to a specific file
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s")
    )
    logger.addHandler(file_handler)


def change_log_file(filename):
    logger = logging.getLogger()
    # Remove all file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    # Add a new file handler with the new filename
    add_file_handler(logger, filename)
