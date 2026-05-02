"""
About this module:
This module contains generic utility functions that are used across different components 
of the machine learning project. These utility functions typically perform common, repetitive tasks 
such as saving or loading files, interacting with databases, or computing generic metrics. 
By centralizing these functions here, we keep our core components clean and adhere to the DRY 
(Don't Repeat Yourself) principle.
"""

import os
import sys

import dill
import numpy as np
import pandas as pd

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves a Python object to disk as a binary file using the dill library.
    
    This is commonly used for saving machine learning models or preprocessing 
    pipelines so they can be loaded later for making predictions.
    
    Args:
        file_path (str): The absolute or relative path where the object should be saved.
        obj (Any): The Python object (e.g., sklearn Pipeline, model) to be saved.
    
    Raises:
        CustomException: If any error occurs during directory creation or file writing.
    """
    try:
        # Extract the directory path from the provided full file path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory (and any necessary parent directories) if they don't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the specified file path in "wb" (write binary) mode
        with open(file_path, "wb") as file_obj:
            # Use dill (which is more robust than pickle) to serialize and save the object
            dill.dump(obj, file_obj)

    except Exception as e:
        # Catch any exception during the saving process, wrap it in our CustomException, and raise it
        raise CustomException(e, sys)
