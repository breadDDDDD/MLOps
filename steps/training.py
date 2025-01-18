import logging
from zenml import step
from sklearn.base import RegressorMixin
import mlflow
import pandas as pd
from model.model_dev import(
    RFmodel,
    GBRTmodel,
    XGBmodel,
    hypertuning
)

@step(enable_cache = False)
def trainingModel(x_train: pd.DataFrame, 
                  x_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,) -> RegressorMixin:
    try:
        model = None
        tuner = None
        model_name ="RF"
        fine_tuning = False
        if model_name == "RF":
            mlflow.sklearn.autolog()
            model = RFmodel()
        elif model_name == "GBRT":
            mlflow.lightgbm.autolog()
            model = GBRTmodel()
        elif model_name == "XGB":
            mlflow.xgboost.autolog()
            model = XGBmodel()
        else:
            raise ValueError("Model not found")
        tuner = hypertuning(model, x_train, y_train, x_test, y_test)

        if fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e