import logging
from zenml import step
import pandas as pd

@step
def trainingData(df: pd.DataFrame) -> None:
    pass
