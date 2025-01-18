import logging
import numpy as np
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from model. data_splitting import DividerStrategy, PreprocessingStrategy, Divider

@step
def cleanData(df:pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame, "x_train"], 
    Annotated[pd.DataFrame, "x_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]]:
    
    try:
        preprocess_strat = PreprocessingStrategy()
        process_cleaning = Divider(df , preprocess_strat)
        preprocessed_data = process_cleaning.handle_data()
        div_strat = DividerStrategy()
        divided_data = Divider(preprocessed_data, div_strat)
        x_train, x_test, y_train, y_test = divided_data.handle_data()
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e   
        
