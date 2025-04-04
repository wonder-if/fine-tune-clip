import json
import os
import logging

# 构建配置文件路径
config_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 向上两层
    "configs",
    "models_info.json",
)


class ModelInfo:
    def __init__(self, model_name, path, config):
        self.model_name = model_name
        self.path = path
        self.config = config

    def __str__(self):
        return f"Model: {self.model_name}, Path: {self.path}, Config: {self.config}"


class ModelManager:
    def __init__(self, models_meta=config_file_path, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self.logger.info("ModelManager initialized")
        self.models = []

        if models_meta is not None:
            self.load_from_json(models_meta)

    def load_from_json(self, file_name):
        self.logger.info(f"Loading models from JSON file: {file_name}")
        try:
            with open(file_name, "r") as f:
                self.models_meta = json.load(f)
            self.models_root = self.models_meta["root"]
            for model_info in self.models_meta["models"]:
                self.logger.debug(f"Processing model: {model_info['name']}")
                self.add_model(
                    model_info["name"], model_info["path"], model_info["config"]
                )
            self.logger.info("Models information loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models from JSON file: {e}")

    # 添加模型
    def add_model(self, model_name, path, config):
        self.logger.info(f"Adding model: {model_name}")

        try:
            full_path = self.models_root + path

            if not os.path.exists(full_path):
                self.logger.error(f"Model path does not exist: {full_path}")
            self.models.append(ModelInfo(model_name, full_path, config))
            self.logger.info(f"Successfully added model{model_name}")
        except Exception as e:
            self.logger.error(f"Failed to add model: {str(e)}")
            raise

    def list_models(self):
        self.logger.info("Listing all models")
        models = []
        for model in self.models:
            models.append(model)
            self.logger.info(model.__str__())
        return models

    def get_model(self, model_name):
        self.logger.info(f"Getting model: {model_name}")

        for model in self.models:
            if model.model_name == model_name:
                self.logger.info(f"Model found: {model_name}")
                return model
        self.logger.warning(f"Model not found: {model_name}")

        return None
