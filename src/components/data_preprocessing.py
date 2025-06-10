import os
import sys
import pandas as pd
import numpy as np
import warnings

from src.logger import logging
from src.exception import SrcException
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from src.config import TARGET_COLUMN

from src.entity import config_entity, artifact_entity

warnings.filterwarnings("ignore")


class DataPreprocessing:
    def __init__(
        self,
        data_preprocessing_config: config_entity.DataPreprocessingConfig,
        feature_engineering_artifact: artifact_entity.FeatureEngineeredArtifact,
    ):
        try:
            logging.info(f'{">"*20} Starting Data Preprocessing {"<"*20}')
            self.data_preprocessing_config = data_preprocessing_config
            self.feature_engineering_artifact = feature_engineering_artifact

        except Exception as e:
            raise SrcException(e, sys)

    def downsample_split(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Downsample the majority class before applying SMOTE.
        """
        try:
            class_counts = df[target].value_counts()
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()

            logging.info(f"Majority class: {majority_class}")
            logging.info(f"Minority class: {minority_class}")
            logging.info(f"Original class distribution:\n{class_counts}")

            # Split into majority and minority
            majority_df = df[df[target] == majority_class]
            minority_df = df[df[target] == minority_class]

            # Downsample majority class
            majority_downsampled = resample(
                majority_df,
                replace=False,
                n_samples=100000,  # Can be adjusted or made configurable
                random_state=42
            )

            balanced_df = pd.concat([majority_downsampled, minority_df])
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

            logging.info(f"Class distribution after downsampling:\n{balanced_df[target].value_counts()}")

            # Train-test split
            train_df, test_df = train_test_split(
                balanced_df,
                stratify=balanced_df[target],
                test_size=0.2,
                random_state=42
            )

            return train_df, test_df

        except Exception as e:
            raise SrcException(e, sys)

    def initiate_data_preprocessing(self) -> artifact_entity.DataPreprocessingArtifact:
        """
        Perform all preprocessing steps and return paths to processed data.
        """
        try:
            logging.info("Step 1: Reading the latest feature-engineered dataset")
            df = pd.read_csv(self.feature_engineering_artifact.feature_engineered_data_file_path)

            # Drop columns not needed
            columns_to_drop = self.data_preprocessing_config.columns_to_drop

            logging.info("Step 2: Balancing classes")
            train_df, test_df = self.downsample_split(df, TARGET_COLUMN)

            logging.info("Step 3: Splitting data and dropping unimportant features")
            X_train = train_df.drop(columns_to_drop, axis=1)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns_to_drop, axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info("Training set before SMOTE:")
            logging.info(y_train.value_counts())

            # Apply SMOTE on training set
            smt = SMOTE(sampling_strategy=0.6, random_state=42)
            X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

            logging.info("Training set after SMOTE:")
            logging.info(pd.Series(y_train_resampled).value_counts())

            logging.info("Test set distribution (unchanged):")
            logging.info(y_test.value_counts())

            logging.info("Creating dataset folder if not available inside data preprocessing")
            preprocessing_dataset_dir=os.path.dirname(self.data_preprocessing_config.train_file_path)
            os.makedirs(preprocessing_dataset_dir , exist_ok=True)

            logging.info("Step 4: Saving training input and target dataset as dataframe")
            train_df=pd.concat([X_train_resampled , y_train_resampled] , axis=1)
            train_df.to_csv(self.data_preprocessing_config.train_file_path , index=False , header=True)

            logging.info("Step 5: Saving testing input and target dataset as dataframe")
            test_df=pd.concat([X_test , y_test] , axis=1)
            test_df.to_csv(self.data_preprocessing_config.test_file_path , index=False , header=True)
            
            data_preprocessing_artifact = artifact_entity.DataPreprocessingArtifact(
                train_file_path=self.data_preprocessing_config.train_file_path,
                test_file_path=self.data_preprocessing_config.test_file_path
            )

            logging.info(f"Step 6: DataPreprocessingArtifact created: {data_preprocessing_artifact}")
            return data_preprocessing_artifact

        except Exception as e:
            raise SrcException(e, sys)
