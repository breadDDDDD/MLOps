import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# abstractmethod
class Eval(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MSE(Eval):
    def calculate_score(self, y_true: np.ndarray, y_pred : np.ndarray)-> float:
        try :
            logging.info("Entered the numebr")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(str(mse))
            return mse
        except Exception as e:
            raise e

class R2(Eval):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        try:
            logging.info("enetered the numbner")
            r2 = r2_score(y_true, y_pred)
            logging.info(str(r2))
            return r2
        except Exception as e:
            raise e
            