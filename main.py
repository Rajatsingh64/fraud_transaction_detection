from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity
from src.components import data_ingestion , data_validation , feature_engineering
import os,sys


if __name__=="__main__":

    try:
        training_pipeline_config=config_entity.TrainingPipelineConfig()

        # Data Ingestion Pipeline
        data_ingestion_config=config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
            )
        data_ingestion_=data_ingestion.DataIngestion(
            data_ingestion_config=data_ingestion_config
        )
        data_ingestion_artifacts=data_ingestion_.initiate_data_ingestion()
        print(f'Data Ingestion Pipeline is Completed Sucessfully')
        logging.info(f'Data Ingestion Pipeline is Completed Sucessfully')

        # Data Validation Pipeline
        data_validation_config=config_entity.DataValidationConfig(
            training_pipeline_config=training_pipeline_config
            )
        data_validation_=data_validation.DataValidation(
            data_ingestion_artifact=data_ingestion_artifacts,
            data_validation_config=data_validation_config
        )
        data_validation_artifacts=data_validation_.initiate_data_validation()
        print(f'Data Validation Pipeline is Completed Sucessfully')
        logging.info(f'Data Validation Pipeline is Completed Sucessfully')

        # Feature Engineering Pipeline
        feature_engineering_config=config_entity.FeatureEngineeringConfig(
                                            training_pipeline_config=training_pipeline_config
        )

        feature_engineering_=feature_engineering.FeatureEngineering(
                                    data_ingestion_artifact=data_ingestion_artifacts,
                                    feature_engineering_config=feature_engineering_config)
        feature_engineering_artifact=feature_engineering_.initiate_feature_engineering()
        print(f'Feature Engineering Pipeline is Completed Sucessfully')
        logging.info(f'Feature Engineering  Pipeline is Completed Sucessfully')

    except Exception as e:
        raise SrcException(e,sys)