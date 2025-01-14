from zenml import pipeline
from steps.ingest_data import ingestDf
from steps.cleaning import cleanData
from steps.training import trainingData
from steps.eval import evalModel

@pipeline
def trainingPipeline(data_path : str):
    df = ingestDf(data_path)
    cleanData(df)
    trainingData(df)
    evalModel(df)
    
   
 

