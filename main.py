from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity
from src.components import (
    data_ingestion,
    data_validation,
    feature_engineering,
    data_preprocessing , 
    model_training
)
import os
import sys


if __name__ == "__main__":
    try:
        # Initialize the overall training pipeline configuration
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # --------------------- Data Ingestion ---------------------
        data_ingestion_config = config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion_ = data_ingestion.DataIngestion(
            data_ingestion_config=data_ingestion_config
        )
        data_ingestion_artifacts = data_ingestion_.initiate_data_ingestion()
        print("Data Ingestion Pipeline completed successfully.")
        logging.info("Data Ingestion Pipeline completed successfully.")

        # --------------------- Data Validation ---------------------
        data_validation_config = config_entity.DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_validation_ = data_validation.DataValidation(
            data_ingestion_artifact=data_ingestion_artifacts,
            data_validation_config=data_validation_config
        )
        data_validation_artifacts = data_validation_.initiate_data_validation()
        print("Data Validation Pipeline completed successfully.")
        logging.info("Data Validation Pipeline completed successfully.")

        # --------------------- Feature Engineering ---------------------
        feature_engineering_config = config_entity.FeatureEngineeringConfig(
            training_pipeline_config=training_pipeline_config
        )
        feature_engineering_ = feature_engineering.FeatureEngineering(
            data_ingestion_artifact=data_ingestion_artifacts,
            feature_engineering_config=feature_engineering_config
        )
        feature_engineering_artifact = feature_engineering_.initiate_feature_engineering()
        print("Feature Engineering Pipeline completed successfully.")
        logging.info("Feature Engineering Pipeline completed successfully.")

        # --------------------- Data Preprocessing ---------------------
        data_preprocessing_config = config_entity.DataPreprocessingConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_preprocessing_ = data_preprocessing.DataPreprocessing(
            data_preprocessing_config=data_preprocessing_config,
            feature_engineering_artifact=feature_engineering_artifact
        )
        data_preprocessing_artifact = data_preprocessing_.initiate_data_preprocessing()
        print("Data Preprocessing Pipeline completed successfully.")
        logging.info("Data Preprocessing Pipeline completed successfully.")

        # --------------------- Model Training ---------------------
        model_training_config = config_entity.ModelTrainingConfig(
            training_pipeline_config=training_pipeline_config
        )
        model_training_ = model_training.ModelTrainer(
            model_training_config=model_training_config,
            data_preprocessing_artifact=data_preprocessing_artifact
        )
        model_training_artifact=model_training_.initiate_model_training()
        print("Model Training Pipeline completed successfully.")
        logging.info("Model Training Pipeline completed successfully.")

        # --------------------- Model Evaluation ---------------------
    
    except Exception as e:
        raise SrcException(e, sys)
