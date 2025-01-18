from zenml.steps import BaseStep

class modelConfig(BaseStep):
    model_name : str = "XGB"
    fine_tuning: bool = False