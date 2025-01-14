import logging
import numpy as np
import pandas as pd
from zenml import step

@step
def cleanData(df:pd.DataFrame)-> None:
    pass
