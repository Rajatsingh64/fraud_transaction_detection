import os
import sys
import pandas as pd
import warnings

from sklearn.metrics import f1_score
from src.logger import logging
from src.exception import SrcException
from src.config import TARGET_COLUMN
from src.predictor import ModelResolver
from src.utils import load_object
from src.entity import config_entity, artifact_entity

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: config_entity.ModelEvaluationConfig,
        model_training_artifact: artifact_entity.ModelTrainingArtifact,
        data_preprocessing_artifact: artifact_entity.DataPreprocessingArtifact,
    ):
        try:
            logging.info(f"{'>'*20} Starting Model Evaluation {'<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.model_training_artifact = model_training_artifact
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SrcException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("Step 1: Reading test dataset as DataFrame")
            test_df = pd.read_csv(self.data_preprocessing_artifact.test_file_path)

            logging.info("Step 2: Splitting test data into features and target")
            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info("Step 3: Loading the previously deployed (production) model")
            previous_model_path = self.model_resolver.get_latest_model_path()
            previous_model = load_object(previous_model_path)

            logging.info("Step 4: Loading the newly trained model")
            current_model = load_object(self.model_training_artifact.model_object_file_path)

            logging.info("Step 5: Evaluating F1 score of the production model")
            previous_pred = previous_model.predict(X_test)
            previous_score = f1_score(y_test, previous_pred)
            logging.info(f"F1 Score (Previous Model): {previous_score}")

            logging.info("Step 6: Evaluating F1 score of the newly trained model")
            current_pred = current_model.predict(X_test)
            current_score = f1_score(y_test, current_pred)
            logging.info(f"F1 Score (Current Model): {current_score}")

            logging.info("Step 7: Comparing current model performance with previous model")
            if current_score < previous_score:
                logging.info("Current model does not outperform the previous model.")
                raise Exception("Current model is not better than the previously deployed model.")

            improved_accuracy = current_score - previous_score
            logging.info(f"Step 8: Improvement in F1 Score: {improved_accuracy}")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_score=improved_accuracy
            )

            logging.info(f"Step 9: Model evaluation artifact created: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise SrcException(e, sys)
