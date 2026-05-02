"""
About this module:
This module handles the Data Ingestion phase of the machine learning pipeline.
Data Ingestion is the process of reading data from various sources (like databases, APIs, or local files)
and preparing it for the next stages of the pipeline (like data transformation and model training). 
Specifically, this script reads the raw dataset, creates a directory for artifacts, saves the raw data,
and splits the data into training and testing sets to be saved as individual CSV files.
"""

import os
import sys
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Determine the project root dynamically to ensure paths work regardless of where the script is executed
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class DataIngestionConfig:
    """
    Configuration class to hold paths related to data ingestion.
    These paths define where the raw, train, and test datasets will be saved.
    """
    train_data_path: str = os.path.join(project_root, "artifacts", "train.csv")
    test_data_path: str = os.path.join(project_root, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(project_root, "artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        # Initialize the configuration to get the file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw dataset, splits it into training and testing sets, and saves them to the artifacts folder.
        Returns the paths to the saved train and test datasets.
        """
        logging.info("Entered the data ingestion method or components.")
        try:
            # Construct the absolute path to the raw dataset and read it into a pandas DataFrame
            dataset_path = os.path.join(
                project_root, "notebook", "data", "students.csv"
            )
            df = pd.read_csv(dataset_path)

            logging.info("Read the dataset as dataframe.")

            # Create the artifacts directory if it doesn't already exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save a copy of the raw dataset into the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")

            # Split the dataset into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            # Save the testing set to a CSV file
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Train test split completed.")

            # Return the file paths of the created train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Catch any error during the ingestion process and raise as a custom exception
            raise CustomException(e, sys)


# Execution block to test the pipeline locally
if __name__ == "__main__":
    # 1. Start the data ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Start the data transformation process using the outputs from data ingestion
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
