import os
import sys

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from transformers import CLIPModel, CLIPTokenizer

from clip_tuner.data import DataManager, get_train_transforms
from clip_tuner.models import ModelManager
from clip_tuner.utils import Logger, get_cfg_defaults, merge_from_file

logger = Logger(log_dir="./logs/", log_file="load_models.log", file_mode="w")

if __name__ == "__main__":
    # cfg = get_cfg_defaults()
    # cfg.merge_from_file("configs/config.yaml")
    # cfg.freeze()
    # logger.info(cfg)

    # Load the model
    model_manager = ModelManager(logger=logger)
    model_manager.list_models()
    model_info = model_manager.get_model("clip-vit-base-patch16")

    model = CLIPModel.from_pretrained(model_info.path)
    tokenizer = CLIPTokenizer.from_pretrained(model_info.path)
    logger.info(tokenizer)
    logger.info(model)
    # Load the data
