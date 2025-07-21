from .model_manager import ModelManager

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPProcessor,
)


def load_model(models_root, models_meta, model_name: str):
    model_manager = ModelManager(models_root, models_meta)
    model_info = model_manager.get_model_info(model_name)
    clip_model = CLIPModel.from_pretrained(model_info.path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_info.path)
    clip_processor = CLIPProcessor.from_pretrained(model_info.path)
    return clip_model, clip_tokenizer, clip_processor
