import os
import sys
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from scipy.stats import uniform, randint

from src.logger import logging
from src.exception import SrcException
from src.config import TARGET_COLUMN
from src.utils import save_object
from src.entity import config_entity, artifact_entity

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Handles training, tuning, evaluation, and saving of the XGBoost model.
    """

    def __init__(
        self,
        model_training_config: config_entity.ModelTrainingConfig,
        data_preprocessing_artifact: artifact_entity.DataPreprocessingArtifact,
        
    ):
        try:
            logging.info(f'{">" * 20} Starting Model Training {"<" * 20}')
            self.model_training_config = model_training_config
            self.data_preprocessing_artifact = data_preprocessing_artifact
        except Exception as e:
            raise SrcException(e, sys)

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Tune XGBoost model using RandomizedSearchCV.
        """
        try:
            pos_count = len(y_train[y_train == 1])
            scale_pos_weight = len(y_train[y_train == 0]) / pos_count if pos_count > 0 else 1.0

            xgb = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=scale_pos_weight
            )

            param_dist = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }

            random_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_dist,
                n_iter=25,
                scoring='f1',
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

            random_search.fit(X_train, y_train)
            logging.info(f"Best Parameters: {random_search.best_params_}")
            return random_search.best_estimator_

        except Exception as e:
            raise SrcException(e, sys)

    def plot_top_features(self, model: XGBClassifier, num_of_features: int, save_path: str) -> pd.DataFrame:
        """
        Plot top N important features using 'gain' as importance type.
        """
        try:
            importance_dict = model.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values(by='importance', ascending=False).head(num_of_features)

            plt.figure(figsize=(10, 6))
            sns.set_style("ticks")
            sns.barplot(data=importance_df, x='importance', y='feature', palette="viridis")
            plt.title('Top XGBoost Feature Importances (Gain)')
            plt.tight_layout()
            plt.savefig(save_path)
            #plt.show()

        except Exception as e:
            raise SrcException(e, sys)

    def precision_recall_performance_plot(self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, save_path: str):
        """
        Generate a precision-recall vs threshold plot.
        """
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

            plt.figure(figsize=(8, 5))
            plt.plot(thresholds, recall[:-1], label='Recall', color='blue')
            plt.plot(thresholds, precision[:-1], label='Precision', color='orange')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision and Recall vs Threshold')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path)
            #plt.show()

        except Exception as e:
            raise SrcException(e, sys)

    def initiate_model_training(self) -> artifact_entity.ModelTrainingArtifact:
        """
        Train, evaluate, and save model and its artifacts.
        """
        try:
            logging.info("Step 1: Reading train/test CSVs")
            train_df = pd.read_csv(self.data_preprocessing_artifact.train_file_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.test_file_path)

            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            logging.info("Step 2: Model training (tuning optional)")
            if getattr(self.model_training_config, "enable_hyperparameter_tuning", False):
                best_model = self.tune_model(X_train, y_train)
            else:
                pos_count = len(y_train[y_train == 1])
                scale_pos_weight = len(y_train[y_train == 0]) / pos_count if pos_count > 0 else 1.0
                best_model = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    scale_pos_weight=scale_pos_weight
                )
                best_model.fit(X_train, y_train)

            logging.info("Step 3: Model evaluation")
            train_f1 = f1_score(y_train, best_model.predict(X_train))
            test_f1 = f1_score(y_test, best_model.predict(X_test))

            logging.info(f"Train F1 Score: {train_f1}")
            logging.info(f"Test F1 Score: {test_f1}")

            logging.info("Classification Report:\n" + classification_report(y_test, best_model.predict(X_test)))

            if test_f1 < self.model_training_config.f1_expected_score:
                raise Exception(f"Model performance below threshold: {test_f1} < {self.model_training_config.f1_expected_score}")

            if abs(train_f1 - test_f1) > self.model_training_config.overfitting_threshold:
                raise Exception(f"Overfitting: F1 diff = {abs(train_f1 - test_f1)}")

            logging.info("Step 4: Saving precision-recall plot")
            logging.info("Creating Model Training Directory if not Available")
            os.makedirs(os.path.dirname(self.model_training_config.top_features_plot_file_path), exist_ok=True)
            
            self.precision_recall_performance_plot(
                model=best_model,
                X_test=X_test,
                y_test=y_test,
                save_path=self.model_training_config.precision_recall_performance_plot_path
            )

            logging.info("Plotting top features")
            self.plot_top_features(best_model, 15, self.model_training_config.top_features_plot_file_path)
            
            logging.info("Step 5: Saving model")
            save_object(self.model_training_config.model_object_file_path, best_model)
            
            model_training_artifact=artifact_entity.ModelTrainingArtifact(
                model_object_file_path=self.model_training_config.model_object_file_path,
                train_f1_score=train_f1,
                test_f1_score=test_f1,
                top_feature_plot_file_path=self.model_training_config.top_features_plot_file_path,
                precision_recall_performance_plot_file_path=self.model_training_config.precision_recall_performance_plot_path
            )
            
            logging.info(f"Step 6: Model Training Artifact Created: {model_training_artifact}")
            return model_training_artifact

        except Exception as e:
            raise SrcException(e, sys)
