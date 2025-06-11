import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity, artifact_entity
from src import utils
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        """
        Initializes the DataIngestion class with config
        """
        try:
            logging.info(f"{'>'*20} Starting Data Ingestion {'<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SrcException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        """
        Performs:
        - Fetching data from MongoDB
        - Storing as CSV in feature store path
        - Returning artifact object with file path
        """
        try:
            logging.info("Step 1: Extracting data from MongoDB into DataFrame...")
            df = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info("Step 2: Creating Feature Store directory if not exists...")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("Step 3: Saving raw data to Feature Store as CSV...")
            df.to_csv(
                path_or_buf=self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True
            )

            logging.info("Step 4: Preparing Data Ingestion Artifact...")
            data_ingestion_artifacts = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )

            logging.info(f"Data Ingestion Artifact Created: {data_ingestion_artifacts}")
            return data_ingestion_artifacts

        except Exception as e:
            raise SrcException(e, sys)
