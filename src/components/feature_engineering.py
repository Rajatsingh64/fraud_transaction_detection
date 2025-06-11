import os
import sys
import pandas as pd
import numpy as np
import warnings

from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity, artifact_entity
from src.feature_extractor import generate_features

warnings.filterwarnings("ignore")


class FeatureEngineering:
    def __init__(
        self,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
        feature_engineering_config: config_entity.FeatureEngineeringConfig,
        
    ):
        """
        Initialize the FeatureEngineering class with ingestion artifact and config.
        """
        try:
            logging.info(f'{">"*20} Starting Feature Engineering {"<"*20}')
            self.data_ingestion_artifact = data_ingestion_artifact
            self.feature_engineering_config = feature_engineering_config

        except Exception as e:
            raise SrcException(e, sys)

    def if_required_columns_exists(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Check whether all required columns exist in the given DataFrame.
        """
        try:
            missing_columns = []

            for column in required_columns:
                if column not in df.columns:
                    logging.info(f'Missing required column: {column}')
                    missing_columns.append(column)

            return len(missing_columns) == 0

        except Exception as e:
            raise SrcException(e, sys)

    def initiate_feature_engineering(self) -> artifact_entity.FeatureEngineeredArtifact:
        """
        Perform feature engineering steps and return the artifact.
        """
        try:
            logging.info("Step 1: Reading dataset from feature store path")
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)

            logging.info("Step 2: Dropping null and duplicate values")
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)

            logging.info("Step 3: Applying feature engineering on the DataFrame")
            logging.info(f"Generating new features using these columns: {self.feature_engineering_config.required_column_names}")
            new_df = generate_features(
                current_df=df,
                past_df=None,  # For now, no past data provided
                required_columns=self.feature_engineering_config.required_column_names
            )

            logging.info("Step 4: Creating feature engineering directory if it doesn't exist")
            feature_engineering_dir = os.path.dirname(self.feature_engineering_config.feature_engineered_data_file_path)
            os.makedirs(feature_engineering_dir, exist_ok=True)

            logging.info("Step 5: Saving the transformed dataset")
            new_df.to_csv(
                self.feature_engineering_config.feature_engineered_data_file_path,
                index=False,
                header=True
            )

            feature_engineering_artifact = artifact_entity.FeatureEngineeredArtifact(
                feature_engineered_data_file_path=self.feature_engineering_config.feature_engineered_data_file_path
            )
            logging.info(f"Step 6: Feature Engineered Artifact Created : {feature_engineering_artifact}")
            return feature_engineering_artifact

        except Exception as e:
            raise SrcException(e, sys)
