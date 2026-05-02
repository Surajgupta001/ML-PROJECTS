"""
About this module:
This module handles the data transformation phase of the machine learning pipeline.
It is responsible for cleaning, encoding, and scaling the data so that it can be fed
into a machine learning model. Specifically, it builds a preprocessing pipeline that
imputes missing values, applies One-Hot Encoding to categorical features, and 
Standard Scaling to numerical features. The final preprocessing object is saved as a 
pickle file to be reused during prediction.
"""

import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Determine the project root dynamically to ensure paths work regardless of where the script is executed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DataTransformationConfig:
    """
    Configuration class to hold paths related to data transformation.
    """
    # Path where the preprocessor object (pickle file) will be saved
    preprocessor_obj_file_path = os.path.join(project_root, "artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        # Initialize the configuration to get the file paths
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the data preprocessing object.
        This function defines the transformation pipelines for both numerical and categorical data.
        """
        try:
            # Define which columns are numerical and which are categorical
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features:
            # 1. Imputer replaces missing values with the median.
            # 2. Scaler standardizes the data (mean=0, variance=1).
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Pipeline for categorical features:
            # 1. Imputer replaces missing values with the most frequent value (mode).
            # 2. OneHotEncoder converts categorical text data into numerical format.
            # 3. Scaler standardizes the encoded data (with_mean=False handles sparse matrices).
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical columns standard scaling pipeline created.")
            logging.info("Categorical columns encoding pipeline created.")

            # Combine both numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            # Catch any error, wrap it in our CustomException, and raise
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test data, applies the preprocessing object, and returns transformed arrays.
        """
        try:
            # Read the CSV files into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info(f"Train DataFrame head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object.")
            # Get the ColumnTransformer object we defined above
            preprocessor_obj = self.get_data_transformer_object()

            # Define the target column (what we want to predict)
            target_column_name = "math_score"
            
            # Separate the input features from the target feature in the training dataset
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate the input features from the target feature in the testing dataset
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Fit the preprocessor on the training data AND transform it
            input_feature_train = preprocessor_obj.fit_transform(input_feature_train_df)
            
            # ONLY transform the testing data (do not fit to prevent data leakage)
            input_feature_test = preprocessor_obj.transform(input_feature_test_df)

            # Concatenate the transformed input features with the target feature into a single array
            # np.c_ stacks the input features and target feature side-by-side
            train_arr = np.c_[input_feature_train, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the fitted preprocessor object to disk so it can be used later for predictions
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            # Return the transformed training array, testing array, and path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Catch any error during the transformation process and raise
            raise CustomException(e, sys)
