import logging
from zenml import step
import numpy as np
import pandas as pd

class ingest_data:
    def __init__(self, data_path : str):
        self.data_path = data_path
    
    def get(self):
        logging.info("ingesting data")
        return pd.read_csv(self.data_path)
@step
def ingestDf(data_path : str) -> pd.DataFrame:
    try:
        ingestData = ingest_data(data_path)
        df = ingestData.get()
        return df
    except Exception as e:
        logging.error("error :{e}")
        raise e