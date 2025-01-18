import logging
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from abc import ABC, abstractmethod
import optuna

# abstract
class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass
    def optimize(self, trial,  x_train, y_train, x_test, y_test):
        pass

class GBRTmodel(Model):
    def train(self, x_train, y_train, **kwargs):
        regression = GradientBoostingRegressor(**kwargs)
        regression.fit(x_train, y_train)
        return regression

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1,200)
        min_sampls_split = trial.suggest_int("min_sample_split", 1 ,4)
        regression = self.train(x_train, y_train, n_estimators = n_estimators, min_sampls_split =min_sampls_split)
        return regression.score(x_test, y_test)
    
class RFmodel(Model):
    def train(self, x_train, y_train, **kwargs):
        regression = RandomForestRegressor(**kwargs)
        regression.fit(x_train, y_train)
        return regression
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        regression = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return regression.score(x_test, y_test)
        
class XGBmodel(Model):
    def train(x_train, y_train, **kwargs):
        regression = XGBRegressor(**kwargs)
        regression.fit(x_train, y_train)
        return regression
    
    def optimize(self, trial ,x_train, y_train, x_test, y_test ):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        regression = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return regression.score(x_test, y_test)
    
class hypertuning:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def optimize(self, n_trial = 100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=100)
        return study.best_trial.params
    