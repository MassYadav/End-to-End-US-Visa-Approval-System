import os, sys
import pandas as pd
from sklearn.metrics import f1_score
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.utils.main_utils import load_object
from us_visa.constants import CURRENT_YEAR, TARGET_COLUMN
from us_visa.entity.estimator import TargetValueMapping


class ModelEvaluation:
    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation...")

            # Load test data (CSV)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Loaded test data from: {self.data_ingestion_artifact.test_file_path}")

            # Add missing derived feature
            if "company_age" not in test_df.columns and "yr_of_estab" in test_df.columns:
                test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]

            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            # Load new trained model
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"Loaded trained model from: {self.model_trainer_artifact.trained_model_file_path}")

            # Load old model if available
            previous_model_path = os.path.join(self.model_eval_config.saved_model_dir, "model.pkl")
            if os.path.exists(previous_model_path):
                old_model = load_object(previous_model_path)
                logging.info(f"Loaded old model from: {previous_model_path}")
            else:
                old_model = None
                logging.info("No previous model found. This model will be accepted by default.")

            # Evaluate both models
            trained_pred = trained_model.predict(X_test)
            trained_score = f1_score(y_test, trained_pred)

            if old_model is not None:
                old_pred = old_model.predict(X_test)
                old_score = f1_score(y_test, old_pred)
            else:
                old_score = 0.0

            # Compare scores
            is_model_accepted = old_model is None or trained_score > old_score + self.model_eval_config.changed_threshold_score

            model_eval_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=trained_score - old_score,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_path=previous_model_path,
            )

            logging.info(f"Model evaluation completed successfully: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise USvisaException(e, sys)
