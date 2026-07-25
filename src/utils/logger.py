import logging
import sys
from pathlib import Path

def get_pipeline_logger(log_file_path: Path, logger_name: str = "ads_marketing_pipeline") -> logging.Logger:
    """
    Initializes and returns a configured named logger that works in both CLI and Airflow.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # False = Logs go only to generate logs.
    # True = Logs go to generated logs and Airflow's UI logs.
    logger.propagate = True

    return logger