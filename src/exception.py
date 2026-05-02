"""
About this module:
This module provides a centralized and detailed custom exception handling system.
Instead of relying on Python's default error messages, which can sometimes be brief,
this module captures the exact file name, line number, and error description when 
an exception is raised. This significantly improves debugging and logging capabilities 
throughout the entire project.
"""

import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information dynamically from the execution trace.
    
    Args:
        error: The caught exception object.
        error_detail (sys): The sys module, used to access execution traceback details.
        
    Returns:
        str: A formatted string containing the exact script name, line number, and error message.
    """
    # exc_info() returns a tuple containing information about the exception being handled.
    # We only need the third element (exc_tb), which holds the traceback object.
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the script/file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the final error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    
    return error_message


class CustomException(Exception):
    """
    A custom exception class that inherits from Python's built-in Exception class.
    It overrides the initialization to automatically capture and format detailed error traces.
    """
    def __init__(self, error_message, error_detail: sys):
        # Initialize the parent Exception class with the original error message
        super().__init__(error_message)
        
        # Generate and store the highly detailed error message using the helper function above
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """
        Overrides the default string representation of the exception.
        When this exception is printed or logged, it outputs our detailed error message.
        """
        return self.error_message