import sys
import os
import logging

def setup_logging(log_file='log.txt'):
    """
    Set up logging to both console and file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        The configured logger
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Create a formatter that includes timestamp, module, and level
    formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    # Create a file handler for writing to log file
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Create a console handler for writing to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger