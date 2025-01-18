import logging
import pandas as pd
from zenml import step
import mlflow
from sklearn.base import RegressorMixin
from model.eval import R2, MSE
from typing_extensions import Annotated
from typing import Tuple



@step
def evalModel(x_test: pd.DataFrame, 
              y_test: pd.Series,
              model : RegressorMixin) -> Tuple[
                Annotated[float, 'r2_score'],
                Annotated[float, 'mse']
              ]:
    try:
        r2_score = R2().calculate_score(y_test, model.predict(x_test))
        mlflow.log_metric("r2_score", r2_score)
        mse = MSE().calculate_score(y_test, model.predict(x_test))
        mlflow.log_metric("mse", mse)
        
        logging.info("The R2 score is: " + str(r2_score))
        logging.info("The MSE score is: " + str(mse))
        return(r2_score,mse)
    except Exception as e:
        logging.error(
            "Exception occurred in evalModel step. Exception message:  " + str(e)
        )
        raise e
