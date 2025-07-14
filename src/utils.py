import os
import sys
import numpy as np
import pandas as pd


from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object):
    """
    This function saves the object to a file.
    """
    try:
        import joblib
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)