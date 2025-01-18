from zenml import pipeline
from steps.ingest_data import ingestDf
from steps.cleaning import cleanData
from steps.training import trainingModel
from steps.eval import evalModel

@pipeline
def trainingPipeline(data_path : str):
    df = ingestDf(data_path)
    x_train, x_test, y_train, y_test = cleanData(df)
    model = trainingModel(x_train,x_test, y_train, y_test)
    r2_score,mse = evalModel(x_test,y_test, model = model)
    
   
 

