import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging

from main.extensions.utils.util import save_object, model_evaluation_reg, model_evaluation_class

@dataclass
class Train_path:
    model_path = os.path.join('main/artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.path = Train_path()
        
        
    def init_train(self):
        try:
            logging.info('Loading datasets')
            X_train = pd.read_csv('main/data/transformed/x_train.csv', index_col=0)
            X_test = pd.read_csv('main/data/transformed/x_test.csv', index_col=0)
            y_train = pd.read_csv('main/data/transformed/y_train.csv')
            y_test = pd.read_csv('main/data/transformed/y_test.csv')
            
            
            model_best_params = {}
            
            model_result = {}
            
            logging.info('Define models and parameters')
            models = {
            "Linear Regressor": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(criterion='squared_error'),
            "Random Forest Regressor": RandomForestRegressor(criterion='absolute_error', n_estimators=128),
            "XGBRegressor": XGBRegressor(n_jobs = -1, learning_rate = 0.05, n_estimators = 128), 
            "AdaBoost Regressor": AdaBoostRegressor(learning_rate=0.05, n_estimators=64),
            'CatBoosting Regressor': CatBoostRegressor(depth=6, iterations=100, learning_rate=0.1),
            'Gradient Boosting': GradientBoostingRegressor(criterion='friedman_mse', learning_rate=0.05, loss = 'squared_error', n_estimators=64, subsample=0.6),
            'K Nearest Neighbors': KNeighborsRegressor(n_neighbors=3),
            'Ridge': Ridge(alpha = 1),
            'Lasso': Lasso(alpha = 0.001)
        }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128]
                },
                "K Nearest Neighbors": {
                    'n_neighbors': [3, 4, 5, 6]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128]
                },
                "Linear Regressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128]
                }
            }
            
            '''
            This part is for Hyperparameter Tuning
            
            for model_name, param in params.items():
                model = GridSearchCV(models[model_name], param, n_jobs=-1, verbose = False)
                model.fit(X_train, y_train)
                model_best_params[model_name] = model.best_params_
                
            return model_best_params
            '''
        
            '''
            This part is for Showing model performance
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, f1 = model_evaluation_class(y_test, y_pred)
                
                model_result[model_name] = accuracy
                model_resut = dict(sorted(model_result.items(), key=lambda item: item[1], reverse=True))
                return model_result  
            
            '''
            logging.info('Model training complete')
            
            model = models['Gradient Boosting']
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            logging.info('Saving model to artifacts')
            save_object(self.path.model_path, model)


        except Exception as e:
            raise CustomException(e, sys)
        
    
    
    