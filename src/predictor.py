import os
from typing import Optional

MODEL_FILE_NAME = "model.pkl"

class ModelResolver:
    """
    A utility class to handle model versioning, path resolution for loading
    and saving machine learning models.
    """

    def __init__(self, model_registry: str = "saved_models", model_dir_name: str = "model"):
        self.model_registry = model_registry
        self.model_dir_name = model_dir_name
        os.makedirs(self.model_registry, exist_ok=True)

    def get_latest_dir_path(self) -> Optional[str]:
        """
        Returns the path to the latest versioned model directory.
        """
        try:
            dir_names = [
                int(name) for name in os.listdir(self.model_registry)
                if name.isdigit()
            ]
            if not dir_names:
                return None
            latest_dir = str(max(dir_names))
            return os.path.join(self.model_registry, latest_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to get latest model directory: {e}")

    def get_latest_model_path(self) -> str:
        """
        Returns the full path to the latest saved model file.
        """
        latest_dir = self.get_latest_dir_path()
        if latest_dir is None:
            raise FileNotFoundError("No existing model found in the registry.")
        return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)

    def get_latest_save_dir_path(self) -> str:
        """
        Determines the next directory path to save a new model version.
        """
        latest_dir = self.get_latest_dir_path()
        next_version = 0 if latest_dir is None else int(os.path.basename(latest_dir)) + 1
        return os.path.join(self.model_registry, str(next_version))

    def get_latest_save_model_path(self) -> str:
        """
        Returns the full path where the next model version should be saved.
        """
        save_dir = self.get_latest_save_dir_path()
        return os.path.join(save_dir, self.model_dir_name, MODEL_FILE_NAME)

class Predictor:
    """
    A wrapper class for handling predictions using the latest available model.
    """

    def __init__(self, model_resolver: ModelResolver):
        self.model_resolver = model_resolver
