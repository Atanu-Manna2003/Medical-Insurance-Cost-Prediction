import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            # Ensure we are looking for the correct directory
            base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the current script's directory
            project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))  # Move up to project root

            model_path = os.path.join(project_root, "artifact", "model.pkl")  # ✅ Corrected path
            preprocessor_path = os.path.join(project_root, "artifact", "preprocessor.pkl")  # ✅ Corrected path

            print(f"Loading model from: {model_path}")  # Debugging log
            print(f"Loading preprocessor from: {preprocessor_path}")  # Debugging log

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self, age: int, sex: str, bmi: float, children: int, smoker: str, region: str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
            "age":[ self.age],
            "sex": [self.sex],
            "bmi": [self.bmi],
            "children": [self.children],
            "smoker": [self.smoker],
            "region": [self.region]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)




        