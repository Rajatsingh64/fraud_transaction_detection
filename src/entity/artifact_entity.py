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
