from dataclasses import dataclass

# ---------------------- Data Ingestion ----------------------
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


# ---------------------- Data Validation ----------------------
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str


# ---------------------- Data Transformation ----------------------
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


# ---------------------- Model Metrics ----------------------
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


# ---------------------- Model Trainer ----------------------
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact


# ---------------------- Model Evaluation ----------------------
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    trained_model_path: str
    best_model_path: str


# ---------------------- Model Pusher ----------------------
@dataclass
class ModelPusherArtifact:
    pusher_model_dir: str   # where new model is stored locally
    saved_model_dir: str    # permanent saved model directory
    model_file_path: str    # actual model.pkl path
