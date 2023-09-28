import os
import sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':

    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    data_transforamtion = DataTransformation()
    train_array, test_array, _ = data_transforamtion.initiate_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_array, test_array)
