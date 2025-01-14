import logging
import pandas as pd
from zenml import step

@step
def evalModel(df: pd.DataFrame) -> None:
    pass
