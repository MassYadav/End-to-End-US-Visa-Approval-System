import os, sys
import shutil
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            # Get the trained model path
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # Define destination for saved model
            saved_model_path = os.path.join(
                self.model_pusher_config.saved_model_dir, "model.pkl"
            )

            os.makedirs(self.model_pusher_config.saved_model_dir, exist_ok=True)

            # Copy model to the saved directory
            shutil.copy(trained_model_path, saved_model_path)
            logging.info(f"Model pushed to {saved_model_path}")

            # Return artifact
            return ModelPusherArtifact(
                saved_model_path=saved_model_path,
                message="Model successfully pushed to production directory."
            )

        except Exception as e:
            raise USvisaException(e, sys)
