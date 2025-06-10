import os
import sys
import pandas as pd
import numpy as np
import warnings

from src.logger import logging
from src.exception import SrcException
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from src.config import TARGET_COLUMN
from sklearn.metrics import precision_recall_curve
from src.utils import save_object
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.entity import config_entity, artifact_entity

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Class responsible for training the XGBoost model,
    tuning hyperparameters, evaluating performance, 
    and saving the model along with feature importance.
    """
    
    def __init__(self, 
                 model_training_config: config_entity.ModelTrainingConfig, 
                 data_preprocessing_artifact: artifact_entity.DataPreprocessingArtifact):
        try:
            logging.info(f'{">"*20} Initializing Model Trainer {"<"*20}')
            self.model_training_config = model_training_config
            self.data_preprocessing_artifact = data_preprocessing_artifact
        except Exception as e:
            raise SrcException(e, sys)

    def tune_model(self, X_train, y_train):
        """
        Tunes the XGBoost model using RandomizedSearchCV.
        """
        try:
            xgb = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))  # handle imbalance
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
            logging.info(f"Best Parameters from Random Search: {random_search.best_params_}")
            return random_search.best_estimator_

        except Exception as e:
            raise SrcException(e, sys)

    def plot_top_features(self, model, num_of_features, save_path):
        """
        Plots and saves the top N important features from the XGBoost model.
        """
        try:
            importance_dict = model.get_booster().get_score(importance_type='weight')
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values(by='importance', ascending=False).head(num_of_features)

            plt.figure(figsize=(10, 6))
            sns.set_style("ticks")
            sns.set_palette("Set2")
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.gca().invert_yaxis()
            plt.xlabel('Feature Importance (Weight)')
            plt.title('Top XGBoost Feature Importances')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

            return importance_df

        except Exception as e:
            raise SrcException(e, sys)
        
    def precision_recall_performance_plot(self,model , X_test ,y_test , save_path):
        
        try:
            # 1. Get predicted probabilities for positive class
            y_scores = model.predict_proba(X_test)[:, 1]

            # 2. Compute precision, recall, thresholds
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

            # 3. Plot Recall vs Threshold
            plt.figure(figsize=(8,5))
            plt.plot(thresholds, recall[:-1], label='Recall', color='blue')
            plt.plot(thresholds, precision[:-1], label='Precision', color='orange')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision and Recall vs Decision Threshold')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.show()

        except Exception as e:
            raise SrcException(e,sys)

    def initiate_model_training(self) -> artifact_entity.ModelTrainingArtifact:
        """
        Trains the XGBoost model, evaluates it, saves the model and top features.
        Returns the model artifact.
        """
        try:
            logging.info("Step 1: Extracting training and testing dataset as Dataframe for Training.")
            train_df=pd.read_csv(self.data_preprocessing_artifact.train_file_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.test_file_path)

            logging.info("Step 2: Splitting training input and target features")
            X_train=train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]
            
            logging.info("Step 3: Splitting testing input and target features")
            X_test= test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            logging.info("Step 4: Training model (hyperparameter tuning optional).")
            # Uncomment below to enable tuning
            # best_model = self.tune_model(X_train, y_train)

            best_model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
            )
            best_model.fit(X_train, y_train)

            logging.info("Step 5: Evaluating model on training data.")
            train_f1 = f1_score(y_train, best_model.predict(X_train))
            logging.info(f"Train F1 Score: {train_f1}")

            logging.info("Step 6: Evaluating model on testing data.")
            test_f1 = f1_score(y_test, best_model.predict(X_test))
            logging.info(f"Test F1 Score: {test_f1}")

            logging.info("Step 7: Validating model performance thresholds.")
            if test_f1 < self.model_training_config.f1_expected_score:
                raise Exception(f"Model performance below expected score: {test_f1} < {self.model_training_config.f1_expected_score}")

            diff = abs(train_f1 - test_f1)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"Overfitting detected. Score diff: {diff} exceeds threshold: {self.model_training_config.overfitting_threshold}")
            logging.info(f"F1 Score difference: {diff}")
            
            logging.info("Step 8: Generating Model Precision and Recall threshold Performance plot.")
            self.precision_recall_performance_plot(self,model=best_model , X_test=X_test , y_test=y_test , save_path=self.model_training_config.precision_recall_performance_plot_path)
            
            logging.info("Step 9: Generating top feature plot.")
            os.makedirs(os.path.dirname(self.model_training_config.top_features_plot_file_path), exist_ok=True)
            important_df = self.plot_top_features(best_model, 15, self.model_training_config.top_features_plot_file_path)
            top_feature_names = important_df["feature"].reset_index(drop=True)

            logging.info("Step 10: Saving model and top features.")
            save_object(self.model_training_config.model_object_file_path, best_model)
            save_object(self.model_training_config.top_features_file_path, top_feature_names)

            logging.info("Step 11: Creating training artifact.")
            return artifact_entity.ModelTrainingArtifact(
                model_object_file_path=self.model_training_config.model_object_file_path,
                train_f1_score=train_f1,
                test_f1_score=test_f1,
                top_features_object_file_path=self.model_training_config.top_features_file_path,
                top_feature_fig_file_path=self.model_training_config.top_features_plot_file_path , 
                precision_recall_performance_plot_file_path=self.model_training_config.precision_recall_performance_plot_path
            )

        except Exception as e:
            raise SrcException(e, sys)
