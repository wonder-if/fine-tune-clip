import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import argparse

from clip_tuner.utils import Logger
from clip_tuner.data import DatasetManager, get_train_transforms
from clip_tuner.utils.config import get_cfg_defaults, merge_from_file

logger = Logger(log_dir="./logs/", log_file="load_datasets.log", file_mode="w")

from collections import Counter
from typing import Union, Optional
from datasets import Dataset, DatasetDict, load_dataset


def count_hf_dataset_classes(
    ds: Union[Dataset, DatasetDict], split: Optional[str] = None
) -> dict:
    """
    统计 Huggingface Dataset 或 DatasetDict 中每个 label 的样本数。

    Args:
        ds: 要统计的 Dataset 或 DatasetDict。
        split: 如果传入的是 DatasetDict，指定要统计的 split 名；
               如果为空且是 DatasetDict，则默认取第一个 split。

    Returns:
        Dict[label_name, count]
    """
    # 如果传入的是 DatasetDict，取出对应 split
    if isinstance(ds, DatasetDict):
        if split is None:
            # 默认取第一个 split
            split = list(ds.keys())[0]
        ds = ds[split]

    # 取出整数编码的 label 列
    labels = ds["label"]
    cnt = Counter(labels)

    # 从 feature 中拿到 id->name 的映射
    feature = ds.features["label"]
    if hasattr(feature, "int2str"):
        mapping = {i: feature.int2str(i) for i in cnt}
    else:
        # 如果不是 ClassLabel，就直接用原始值
        mapping = {i: str(i) for i in cnt}

    return {mapping[i]: c for i, c in cnt.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for image classification"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/base_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify configuration from command line"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting main program")

    cfg = get_cfg_defaults()
    if args.config_file:
        merge_from_file(cfg, args.config_file)
    cfg.freeze()

    logger.info("Configuration:\n{}".format(cfg))

    data_manager = DatasetManager("./configs/datasets_info.json", logger=logger)
    logger.info("Data manager initialized, available datasets:")
    # logger.info(data_manager.list_datasets())
    train_transforms = get_train_transforms()
    for data_info in data_manager.list_datasets():
        dataset = data_manager.load_hf_dataset(
            data_info.dataset_name, data_info.domain_name
        )
        dataset.set_transform(train_transforms)
        logger.info(dataset)
    logger.info(dataset)
