import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from clip_tuner.data import DatasetManager, get_val_transforms, zero_shot_collate_fn
from clip_tuner.models import ModelManager
from clip_tuner.trainers import BaseTrainer
from clip_tuner.utils import Logger, get_logger


def run_zero_shot_baseline(
    model_name="clip-vit-base-patch32", datasets_cfg="./configs/datasets_info.json"
):
    # 初始化模型
    model_info = ModelManager().get_model(model_name)
    model = CLIPModel.from_pretrained(model_info.path)
    processor = CLIPProcessor.from_pretrained(model_info.path)
    trainer = BaseTrainer(model=model)

    results = []
    data_manager = DatasetManager(datasets_cfg, logger=None)
    for info in data_manager.list_datasets():
        ds = data_manager.load_hf_dataset(info.dataset_name, info.domain_name)["train"]
        ds.set_transform(
            get_val_transforms(
                image_size=(
                    processor.image_processor.crop_size["height"],
                    processor.image_processor.crop_size["width"],
                ),
                image_mean=processor.image_processor.image_mean,
                image_std=processor.image_processor.image_std,
            )
        )
        metrics = trainer.zero_shot_evaluate(
            ds,
            data_collator=zero_shot_collate_fn,
            processor=processor,
        )
        results.append(
            {
                "dataset": info.dataset_name,
                "domain": info.domain_name,
                **metrics,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("zero_shot_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    run_zero_shot_baseline()
