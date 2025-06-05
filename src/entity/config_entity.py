import os
import sys
import logging
from datetime import datetime
from src.exception import SrcException 
from src.config import database_name 

class TrainingPipelineConfig:
    def __init__(self):
        try:
            logging.info(f"{'>' * 20} Training Pipeline(Required Python 3.12 version){'<' * 20}")
            
            # Generate timestamp for unique artifact folder name
            datetime_file_name = datetime.now().strftime("%d%m%y__%H%M%S")
            
            # Create artifact directory path inside the current working directory
            self.artifact_directory = os.path.join(os.getcwd(),'artifacts', datetime_file_name)
        
        except Exception as e:
            raise SrcException(e, sys)
        
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
      
        # Database and collection settings
        self.database_name = database_name
        self.collection_name = "transactions"

        # Base data ingestion directory
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_directory,
            "data_ingestion"
        )

        # Feature store path
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            "feature_store",
            "main.csv"
        )

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Configuration class for data validation.
        Sets paths for validation reports and base dataset,
        and defines thresholds for validation checks.
        """

        # Directory to store data validation artifacts
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_directory,
            'data_validation'
        )

        # Path to the generated validation report
        self.report_file_path = os.path.join(
            self.data_validation_dir,
            'report.yml'
        )

        # Threshold to flag missing columns (20% or more missing = drop/alert)
        self.missing_columns_threshold = 0.2

        # Reference file to compare current dataset against
        self.base_file_path = os.path.join("dataset", "main.csv") # Previous production dataset ,using latest if not available

class FeatureEngineeringConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to store feature-engineered artifacts
        self.feature_engineering_dir = os.path.join(
            training_pipeline_config.artifact_directory, 'feature_engineered'
        )

        # Path to the final feature-engineered dataset
        self.feature_engineered_data_file_path = os.path.join(
            self.feature_engineering_dir, "feature_engineered_main.csv"
        )

        # List of required column names for downstream tasks
        self.required_column_names = ['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_DATETIME', 'TX_FRAUD']
