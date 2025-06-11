import os
import sys
import pandas as pd
import warnings

from src.logger import logging
from src.exception import SrcException
from src.predictor import ModelResolver
from src.utils import load_object, save_object
from src.entity import config_entity, artifact_entity

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

class ModelPusher:
    def __init__(self, 
                 model_pusher_config: config_entity.ModelPusherConfig,
                 model_training_artifact: artifact_entity.ModelTrainingArtifact,
                 model_evaluation_artifact: artifact_entity.ModelEvaluationArtifact):
        """
        Initializes the ModelPusher with required configs and artifacts.
        """
        try:
            logging.info(f"{'>' * 20} Starting Model Pusher {'<' * 20}")
            self.model_pusher_config = model_pusher_config
            self.model_training_artifact = model_training_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SrcException(e, sys)

    def initiate_model_pusher(self) -> artifact_entity.ModelPusherArtifact:
        """
        Handles pushing the trained model to both the pusher directory and versioned saved_models directory.
        Returns:
            ModelPusherArtifact containing both directory paths.
        """
        try:
            # --------------------------------------------------------------------
            # Step 1: Load current trained model
            # --------------------------------------------------------------------
            logging.info("Step 1: Loading the current trained model")
            trained_model = load_object(self.model_training_artifact.model_object_file_path)
            logging.info("Successfully loaded the current trained model.")

            # --------------------------------------------------------------------
            # Step 2: Save model to pusher directory
            # --------------------------------------------------------------------
            logging.info("Step 2: Saving the model to the model pusher directory")
            save_object(
                file_path=self.model_pusher_config.pusher_model_file_path,
                obj=trained_model
            )
            logging.info(f"Model successfully saved")

            # --------------------------------------------------------------------
            # Step 3: Save model to saved_models directory using ModelResolver
            # --------------------------------------------------------------------
            logging.info("Step 3: Saving the model to the versioned saved_models directory")
            saved_model_path = self.model_resolver.get_latest_save_model_path()
            save_object(
                file_path=saved_model_path,
                obj=trained_model
            )
            logging.info(f"Model successfully saved")

            # --------------------------------------------------------------------
            # Step 4: Create and return ModelPusherArtifact
            # --------------------------------------------------------------------
            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Step 4: ModelPusherArtifact created: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise SrcException(e, sys)
