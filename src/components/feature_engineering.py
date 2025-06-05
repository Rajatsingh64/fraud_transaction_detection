import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import SrcException
from src.entity import config_entity, artifact_entity
from src.feature_extractor import generate_features
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineering:

    def __init__(self,data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 feature_engineering_config:config_entity.FeatureEngineeringConfig):

        try:
            logging.info(f'{">"*20} Feature Engineering {"<"*20} ')
            self.data_ingestion_artifact=data_ingestion_artifact
            self.feature_engineering_config=feature_engineering_config
        
        except Exception as e:
            raise SrcException(e,sys)
        

    def if_required_columns_exists(self,df:pd.DataFrame , required_columns:list)->bool:
        
        try:
            missing_columns=[]
            for column in required_columns:
               if column not in df.columns:
                   logging.info(f'Required Column {column} not found in dataframe')
                   missing_columns.append(column)
            if missing_columns:
                return False
            
            return True
                   
        except Exception as e:
            raise SrcException(e,sys)


    def initiate_feature_engineering(self)->artifact_entity.FeatureEngineeredArtifact:

        try:
            logging.info(f'Step 1: Reading Current Dataset as Dataframe')
            df=pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)

            logging.info(f'Step 2: Droping Null and Duplicated Values from Current Dataframe')
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            
          
            logging.info(f'Applying Feature Engineering to a DataFrame')
            new_df=generate_features(current_df=df,
                                     past_df=None,
                                     required_columns=self.feature_engineering_config.required_column_names)
            
            new_df.to_csv(
                self.feature_engineering_config.feature_engineered_data_file_path,
                  index=False, 
                  header=True)
            
            logging.info(f'Creating Feature Engineering Artifact')
            feature_engineering_artifact=artifact_entity.FeatureEngineeredArtifact(
                feature_engineered_data_file_path=self.feature_engineering_config.feature_engineered_data_file_path
            )

            return feature_engineering_artifact
           
            
            


        except Exception as e:
            raise SrcException(e,sys)