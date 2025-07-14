import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'

    def predict(self, features: pd.DataFrame):
        """
        Load preprocessor and model, transform features, predict.
        """
        try:
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Transform input data using preprocessor
            logging.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)
            logging.info(f"Prediction completed: {preds}")

            return preds  # Return predictions array

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Converts user input to DataFrame.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            logging.info(f"Custom data input dict: {custom_data_input_dict}")
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error(f"Error in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)
        logging.info("Custom data converted to DataFrame successfully")
        