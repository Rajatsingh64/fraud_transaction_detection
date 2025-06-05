from dataclasses import dataclass
import os,sys

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str
 
@dataclass
class FeatureEngineeredArtifact:
    feature_engineered_data_file_path:str