import os
import logging

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPProcessor,
)


class ModelInfo:
    def __init__(self, model_name, path, config):
        self.model_name = model_name
        self.path = path
        self.config = config

    def __str__(self):
        return f"Model: {self.model_name}, Path: {self.path}, Config: {self.config}"


class ModelManager:
    def __init__(self, models_root, models_meta):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("ModelManager initialized")
        self.models = []
        self.models_root = models_root
        self.models_meta = models_meta
        self.load_from_config()

    def load_from_config(self):
        for model_meta in self.models_meta:
            self.logger.debug(f"Processing model: {model_meta['name']}")
            self.add_model(model_meta["name"], model_meta["path"], model_meta["config"])
        self.logger.debug("Models information loaded successfully")

    # 添加模型
    def add_model(self, model_name, path, config):
        self.logger.debug(f"Adding model: {model_name}")

        try:
            full_path = self.models_root + path

            if not os.path.exists(full_path):
                self.logger.error(f"Model path does not exist: {full_path}")
            self.models.append(ModelInfo(model_name, full_path, config))
            self.logger.debug(f"Successfully added model{model_name}")
        except Exception as e:
            self.logger.error(f"Failed to add model: {str(e)}")
            raise

    def list_models(self):
        self.logger.debug("Listing all models")
        models = []
        for model in self.models:
            models.append(model)
            self.logger.info(model.__str__())
        return models

    def get_model_info(self, model_name):
        self.logger.debug(f"Getting model: {model_name}")

        for model in self.models:
            if model.model_name == model_name:
                self.logger.debug(f"Model found: {model_name}")
                return model
        self.logger.warning(f"Model not found: {model_name}")

        return None


def load_model(models_root, models_meta, model_name: str):
    model_manager = ModelManager(models_root, models_meta)
    model_info = model_manager.get_model_info(model_name)
    clip_model = CLIPModel.from_pretrained(model_info.path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_info.path)
    clip_processor = CLIPProcessor.from_pretrained(model_info.path)
    return clip_model, clip_tokenizer, clip_processor
