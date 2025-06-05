import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from src.utils import write_yaml_file
from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity, artifact_entity
import warnings
warnings.filterwarnings("ignore")


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_validation_config: config_entity.DataValidationConfig):
        try:
            logging.info(f"{'>'*20} Data Validation Initialized {'<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.validation_error = dict()
        except Exception as e:
            raise SrcException(e, sys)
        
    def if_required_columns_exists(self,
                                    base_df: pd.DataFrame,
                                    current_df: pd.DataFrame,
                                    report_key_name: str) -> bool:
        """
        Verifies that all required columns in the base dataset exist in the current dataset.
        Logs and records any missing columns.
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = [col for col in base_columns if col not in current_columns]

            if missing_columns:
                self.validation_error[report_key_name] = missing_columns
                logging.warning(f"Missing required columns: {missing_columns}")
                return False

            logging.info("All required columns exist in the current dataset.")
            return True

        except Exception as e:
            raise SrcException(e, sys)
    
    def check_data_drift(self,
                         base_df: pd.DataFrame,
                         current_df: pd.DataFrame,
                         report_key_name: str):
        """
        Performs data drift detection between base and current datasets.

        - For numeric and datetime-derived columns: uses the KS test.
        - For categorical columns: uses the Chi-square test.
        """
        try:
            drift_report = {}

            for column in base_df.columns:
                base_data = base_df[column]
                current_data = current_df[column]

                # Detect if the column is datetime
                if base_data.dtype == 'O' and pd.to_datetime(base_data, errors='coerce').notna().all():
                    # Convert string to datetime
                    base_data = pd.to_datetime(base_data, errors="coerce")
                    current_data = pd.to_datetime(current_data, errors="coerce")

                    # Extract datetime features
                    base_date_features = {
                        "year": base_data.dt.year,
                        "month": base_data.dt.month,
                        "weekday": base_data.dt.weekday,
                        "day": base_data.dt.day,
                        "is_weekend": base_data.dt.weekday >= 5
                    }

                    current_date_features = {
                        "year": current_data.dt.year,
                        "month": current_data.dt.month,
                        "weekday": current_data.dt.weekday,
                        "day": current_data.dt.day,
                        "is_weekend": current_data.dt.weekday >= 5
                    }

                    # Apply KS test on each derived feature
                    for feature in base_date_features:
                        base_feature = base_date_features[feature]
                        current_feature = current_date_features[feature]

                        stat = ks_2samp(base_feature, current_feature)
                        drift_report[f"{column}_{feature}"] = {
                            "pvalue": float(stat.pvalue),
                            "Same_distribution": bool(stat.pvalue > 0.05)
                        }
                
                # For numeric columns
                elif base_data.dtype in [np.float64, np.int64]:
                    logging.info(f"Running KS test on numeric column: {column}")
                    stat = ks_2samp(base_data, current_data)
                    drift_report[column] = {
                        "pvalue": float(stat.pvalue),
                        "Same_distribution": bool(stat.pvalue > 0.05)
                    }

                # For categorical columns
                else:
                    try:
                        contingency_table = pd.crosstab(base_data, current_data)
                        chi2, p_value, _, _ = chi2_contingency(contingency_table)

                        drift_report[column] = {
                            "pvalue": float(p_value),
                            "Same_distribution": bool(p_value > 0.05)
                        }
                        
                    except Exception:
                        logging.warning(f"Skipping drift check for '{column}' due to insufficient overlapping categories.")

            # Store the drift report
            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise SrcException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        """
        Orchestrates the entire data validation process:
        - Loads datasets
        - Validates required columns
        - Performs drift detection
        - Writes validation report to a YAML file
        """
        try:
            logging.info('Step 1: Loading base and current datasets.')
            base_df = pd.read_csv(self.data_validation_config.base_file_path).dropna()
            current_df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path).dropna()

            logging.info('Step 2: Checking if required columns exist in the current dataset.')
            columns_valid = self.if_required_columns_exists(
                base_df=base_df,
                current_df=current_df,
                report_key_name="Missing_columns_within_current_main_dataset"
            )

            if columns_valid:
                logging.info('Step 3: Performing data drift detection.')
                self.check_data_drift(
                    base_df=base_df,
                    current_df=current_df,
                    report_key_name="Data_drift_within_current_dataset"
                )

            logging.info('Step 4: Saving data validation report to file.')
            write_yaml_file(
                file_path=self.data_validation_config.report_file_path,
                data=self.validation_error
            )

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path
            )

            logging.info(f"Step 5: DataValidationArtifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise SrcException(e, sys)
