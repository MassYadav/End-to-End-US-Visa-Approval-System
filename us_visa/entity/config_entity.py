import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

# ---------------------- Timestamp ----------------------
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# ---------------------- Training Pipeline Config ----------------------
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# ---------------------- Data Ingestion ----------------------
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# ---------------------- Data Validation ----------------------
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_DRIFT_REPORT_DIR,
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
    )


# ---------------------- Data Transformation ----------------------
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy")
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME.replace("csv", "npy")
    )
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCSSING_OBJECT_FILE_NAME
    )


# ---------------------- Model Trainer ----------------------
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_FILE_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


# ---------------------- Model Evaluation ----------------------
@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
    report_file_path: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_REPORT_FILE_NAME)
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

    trained_model_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_FILE_NAME
    )

    saved_model_dir: str = MODEL_PUSHER_SAVED_MODEL_DIR  # where old/best model is stored
    target_column: str = TARGET_COLUMN
    current_year: int = CURRENT_YEAR


# ---------------------- Model Pusher ----------------------
@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME)
    saved_model_dir: str = MODEL_PUSHER_SAVED_MODEL_DIR
    pusher_model_dir: str = os.path.join(model_pusher_dir, MODEL_PUSHER_SAVED_MODEL_DIR)


# ---------------------- Target Value Mapping ----------------------
class TargetValueMapping:
    def __init__(self):
        # Label encoding for visa approval prediction
        self.mapping = {
            "Approved": 1,
            "Denied": 0
        }

    def to_dict(self):
        """Return dictionary form of mapping."""
        return self.mapping

    def reverse_mapping(self):
        """Return reverse mapping (numeric → label)."""
        return {v: k for k, v in self.mapping.items()}

