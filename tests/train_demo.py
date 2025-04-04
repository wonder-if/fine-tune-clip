import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import argparse
from datasets import load_dataset

from clip_tuner.utils import Logger
from clip_tuner.data import DataManager, get_train_transforms
from clip_tuner.utils.config import get_cfg_defaults, merge_from_file

logger = Logger(log_dir="./logs/", log_file="load_datasets.log", file_mode="w")


def parse_args():
    """ """
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

    data_manager = DataManager("./configs/datasets_info.json", logger=logger)
    data_info = data_manager.get_dataset("office-31", "webcam")
    dataset = load_dataset(data_info.data_dir)
    train_transforms = get_train_transforms()
    dataset.set_transform(train_transforms)
    logger.info(dataset)

    logger.info("Main program completed")
