import os
import sys
import logging
from datetime import datetime
from src.exception import SrcException 
from src.config import database_name 


class TrainingPipelineConfig:
    def __init__(self):
        try:
            logging.info(f"{'>' * 20} Training Pipeline (Requires Python 3.12) {'<' * 20}")
            
            # Generate timestamp-based folder name for artifacts
            datetime_file_name = datetime.now().strftime("%d%m%y__%H%M%S")
            
            # Path to store all artifacts
            self.artifact_directory = os.path.join(os.getcwd(), 'artifacts', datetime_file_name)
        
        except Exception as e:
            raise SrcException(e, sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # MongoDB connection configuration
        self.database_name = database_name
        self.collection_name = "transactions"

        # Data ingestion directory
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_directory, "data_ingestion"
        )

        # File path to store the raw dataset
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir, "feature_store", "main.csv"
        )


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Configuration class for data validation.
        Defines where to store the validation report, threshold for missing values,
        and the reference file for comparison.
        """

        # Directory to store data validation outputs
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_directory, "data_validation"
        )

        # Path for the data validation report
        self.report_file_path = os.path.join(
            self.data_validation_dir, "report.yml"
        )

        # Drop or alert if a column has more than 20% missing values
        self.missing_columns_threshold = 0.2

        # Baseline dataset path (for schema comparison)
        self.base_file_path = os.path.join("dataset", "main.csv")


class FeatureEngineeringConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for storing feature-engineered files
        self.feature_engineering_dir = os.path.join(
            training_pipeline_config.artifact_directory, "feature_engineered"
        )

        # Path for the final feature-engineered dataset
        self.feature_engineered_data_file_path = os.path.join(
            self.feature_engineering_dir, "feature_engineered_main.csv"
        )

        # Required columns after feature engineering
        self.required_column_names = [
            'TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID',
            'TX_AMOUNT', 'TX_DATETIME', 'TX_FRAUD'
        ]


class DataPreprocessingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to store preprocessed data
        self.data_preprocessing_dir = os.path.join(
            training_pipeline_config.artifact_directory, "data_preprocessing"
        )

        # Paths to save processed training and test data as NumPy arrays
        self.train_file_path = os.path.join(
            self.data_preprocessing_dir, "dataset", "train.csv"
        )

        self.test_file_path = os.path.join(
            self.data_preprocessing_dir, "dataset", "test.csv"
        )

        # Columns to exclude from model training
        self.columns_to_drop = [
            'TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID',
            'TX_AMOUNT', 'TX_DATETIME', 'TX_FRAUD'
        ]

class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig, enable_hyperparameter_tuning: bool = False):
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_directory, "model_training")

       # Toggle for hyperparameter tuning
        self.enable_hyperparameter_tuning=True
        
        # Paths for saving model, features, and plots
        self.model_object_file_path = os.path.join(self.model_training_dir, "model.pkl")
        self.top_features_file_path = os.path.join(self.model_training_dir, "trained_features", "top_features.pkl")
        self.top_features_plot_file_path = os.path.join(self.model_training_dir, "trained_features", "top_features.png")
        self.precision_recall_performance_plot_path = os.path.join(self.model_training_dir, "precision_recall_performance.png")

        # Performance constraints
        self.f1_expected_score = 0.8
        self.overfitting_threshold = 0.1

