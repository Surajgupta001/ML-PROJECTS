"""
About this module:
This module sets up a centralized logging system for the entire application.
Instead of using standard `print()` statements, the project uses this logger to track 
execution flows, record data transformations, and log critical errors. All logs are 
automatically timestamped and saved into a dedicated `logs/` directory at the project root,
allowing for easy monitoring and debugging.
"""

import logging
import os
from datetime import datetime


# Generate a dynamic log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m-%d_%Y_%H-%M-%S')}.log"

# Dynamically determine the root directory of the project 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the path for the generic logs directory at the root level
logs_path = os.path.join(project_root, "logs")

# Create the logs directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Combine the directory path and the dynamic file name to get the absolute path for the new log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# Configure the global logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    # Define the exact format of every log message:
    # [Timestamp] LineNumber LoggerName - LogLevel - Actual Message
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    # Set the minimum severity level to log (INFO catches standard processes, warnings, and errors)
    level=logging.INFO,
)