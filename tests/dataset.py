import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from clip_tuner.utils import Logger
from clip_tuner.data import DataManager, get_train_transforms
from clip_tuner.utils.config import get_cfg_defaults, merge_from_file

logger = Logger(file_mode="a")


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


def visualize_dataset(dataset, num_samples=5):
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建一个数据加载器
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # 获取一批数据

    example_batch = next(iter(dataloader))
    images = example_batch["pixel_values"]
    labels = example_batch["label"]
    # 可视化图像
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        image = images[i].numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean  # 反归一化
        image = np.clip(image, 0, 1)  # 限制像素值在 [0, 1]
        plt.imshow(image)
        plt.title(f"Label: {labels[i].item()}")
        plt.tight_layout()
        plt.axis("off")
    plt.savefig("./logs/figs/plt.png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting main program")

    # 加载配置
    cfg = get_cfg_defaults()
    if args.config_file:
        merge_from_file(cfg, args.config_file)
    cfg.freeze()
    logger.info("Configuration:\n{}".format(cfg))

    # 初始化数据管理器
    data_manager = DataManager("./configs/datasets_info.json")
    data_manager.list_datasets()
    data_info = data_manager.get_dataset("office-31", "webcam")

    # 加载数据集
    dataset = load_dataset(data_info.data_dir)["train"]

    # 获取数据增强转换
    train_transforms = get_train_transforms(cfg)

    # 应用数据增强
    dataset.set_transform(train_transforms)

    # 打印数据集信息
    logger.info("Dataset loaded successfully")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset features: {dataset.features}")

    # 可视化数据样本
    logger.info("Visualizing dataset samples")
    visualize_dataset(dataset)

    # 检查数据格式
    logger.info("Checking data format")
    example = dataset[0]
    logger.info(f"Image shape: {example['pixel_values'].shape}")
    logger.info(f"Label: {example['label']}")

    logger.info("Main program completed")
