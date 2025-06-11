from dataclasses import dataclass
import os
import sys

@dataclass
class DataIngestionArtifact:
    """
    Stores the path of the file generated after data ingestion.
    """
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Stores the path to the data validation report.
    """
    report_file_path: str


@dataclass
class FeatureEngineeredArtifact:
    """
    Stores the path to the feature-engineered dataset.
    """
    feature_engineered_data_file_path: str


@dataclass
class DataPreprocessingArtifact:
    """
    Stores the paths to training and testing datasets after preprocessing.
    """
    train_file_path: str
    test_file_path: str

@dataclass
class ModelTrainingArtifact:
    """
    Stores the paths to trained model and top features after model training.
    """
    model_object_file_path: str
    train_f1_score: float
    test_f1_score: float
    precision_recall_performance_plot_file_path: str
    top_features_object_file_path: str
    top_feature_plot_file_path: str

