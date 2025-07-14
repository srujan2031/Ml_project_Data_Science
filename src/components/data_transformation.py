import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates a data transformation pipeline.
        """
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education",
                                   "lunch", "test_preparation_course"]
            
            logging.info("Creating numerical and categorical transformer pipelines")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline with FIX applied (with_mean=False)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder()),  # sparse output
                ('scaler', StandardScaler(with_mean=False))  # FIX here ✅
            ])

            # Combine numerical and categorical pipelines
            logging.info("Creating ColumnTransformer object")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Data transformation pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process.
        """
        try:
            logging.info("Reading train and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Debug columns
            logging.info(f"Train Data Columns: {train_df.columns.tolist()}")
            logging.info(f"Test Data Columns: {test_df.columns.tolist()}")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Data transformation completed successfully")
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
