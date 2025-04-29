import pandas as pd
import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from dataclasses import dataclass
from main.extensions.utils.util import load_object

@dataclass
class ModelPath:
    model_path = 'main/artifacts/regression/model.pkl'
    preprocessor_path = 'main/artifacts/regression/processor.pkl'
    
class Prediction:
    def __init__(self):
        self.path = ModelPath()
        
    def init_predict(self, data):
        try:
            model = load_object(self.path.model_path)
            processor = load_object(self.path.preprocessor_path)

            data_transformed = processor.transform(data)
            
            pred = model.predict(data_transformed)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
          
class Data:
    def __init__(self, area: float, bedrooms: int, property_type: str, 
                 furniture: str, legal_status: str, distance_to_center: float, bathrooms: int, floors: int, name: str, age: int, review: str, score: int):
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.floors = floors
        self.property_type = property_type
        self.furniture = furniture
        self.legal_status = legal_status
        self.distance_to_center = distance_to_center
        self.name = name
        self.age = age
        self.review = review
        self.score = score

    def get_user(self):
        data = {
            'name': self.name,
            'age': self.age
        }
        return data
    
    def get_review(self):
        return self.review
    
    def get_data(self):
        data_input = {
            'area': [self.area],
            'bedrooms': [self.bedrooms],
            'bathrooms': [self.bathrooms],
            'floors': [self.floors],
            'property_type': [self.property_type],
            'furniture': [self.furniture],
            'legal_status': [self.legal_status],
            'distance_to_center': [self.distance_to_center]
        }
        return pd.DataFrame(data_input)

            
    