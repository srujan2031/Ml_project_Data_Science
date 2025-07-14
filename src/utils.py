import os
import sys
import numpy as np
import pandas as pd


from src.exception import CustomException
from src.logger import logging

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    report = {}
    for name, model in models.items():
        try:
            print(f"Tuning hyperparameters for {name}...")
            params = param.get(name, {})
            if params:
                gs = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[name] = score
            print(f"{name}: R²={score:.4f}")

        except Exception as e:
            raise CustomException(e, sys)
    return report
