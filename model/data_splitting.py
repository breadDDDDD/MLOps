import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Union

# abstract
class strategy_data(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        pass


class PreprocessingStrategy(strategy_data):
    def handle_data(self, data : pd.DataFrame)-> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix",
                    "order_item_id"
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            return data
        
        except Exception as e:
            logging.error(e)
            raise e

class DividerStrategy(strategy_data):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e

class Divider:
    def __init__(self, data: pd.DataFrame, strategy: strategy_data) -> None:
        self.df = data
        self.strategy = strategy

    def handle_data(self)-> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.df)