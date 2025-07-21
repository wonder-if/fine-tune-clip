import os
import json
import logging

from datasets import (
    DatasetDict,
    Features,
    ClassLabel,
    Image as HfImage,
    load_from_disk,
)

# 构建配置文件路径
config_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 向上两层
    "configs",
    "datasets_info.json",
)


class DatasetInfo:
    # 定义一个数据集信息类
    def __init__(
        self, dataset_name, domain_name, data_dir, num_classes, class_names=None
    ):
        self.dataset_name = dataset_name
        # 数据集名称
        self.domain_name = domain_name
        # 数据集所属领域
        self.data_dir = data_dir
        # 数据集存放路径
        self.num_classes = num_classes
        # 数据集类别数量
        self.class_names = class_names

    def __str__(self):
        return f"DatasetInfo(dataset_name={self.dataset_name}, domain_name={self.domain_name}, data_dir={self.data_dir}, num_classes={self.num_classes})"


"""
    a. 数据集管理器：
        类图：
            DatasetManager
                - __init__: 初始化数据集管理器
                    - datasets_meta: 数据集信息
                    - logger: 日志记录器
                    
                - load_from_json: 从JSON文件中加载所有数据集信息
                - add_dataset: 添加数据集信息
                - list_datasets: 列出所有数据集信息
        
                
"""


class DatasetManager:
    """
    数据集管理器，用于管理多个数据集

    Attributes:
        datasets (list): 存储所有数据集的列表
        logger (logging.Logger): 日志记录器

    Methods:
        __init__(self, datasets_meta, dataset_root=None, logger=None): 初始化数据集管理器
        load_from_json(self, file_name, dataset_root=None): 从JSON文件中加载所有数据集信息
        add_dataset(self, dataset_name, data_dir, num_classes, domain_info=None): 添加数据集信息
        list_datasets(self): 列出所有数据集信息
    """

    def __init__(self, datasets_root, datasets_meta):
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataManager initialized")
        self.datasets_root = datasets_root
        self.datasets_meta = datasets_meta
        self.datasets = []
        self.load_from_config()

    def load_from_config(self):
        try:
            for dataset_meta in self.datasets_meta:
                self.logger.debug(f"Processing dataset: {dataset_meta['name']}")
                if dataset_meta["domains"] is not None:
                    for domain_info in dataset_meta["domains"]:
                        self.add_dataset(
                            dataset_meta["name"],
                            dataset_meta["dir"],
                            dataset_meta["num_classes"],
                            domain_info,
                        )
                else:
                    self.add_dataset(
                        dataset_meta["name"],
                        dataset_meta["dir"],
                        dataset_meta["num_classes"],
                        None,
                    )
            self.logger.info("Datasets information loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading datasets from JSON file: {e}")
            raise

    # 添加数据集
    def add_dataset(self, dataset_name, data_dir, num_classes, domain_info):
        self.logger.info(
            f"Adding dataset: {dataset_name}, Domain: {domain_info['domain_name']}"
        )
        if domain_info is not None:
            try:
                data_dir = self.datasets_root + data_dir + domain_info["domain_dir"]

                class_names = None
                if os.path.exists(data_dir):
                    class_names = os.listdir(data_dir)
                else:
                    self.logger.error(f"Data directory does not exist: {data_dir}")
                self.datasets.append(
                    DatasetInfo(
                        dataset_name,
                        domain_info["domain_name"],
                        data_dir,
                        num_classes,
                        class_names,
                    )
                )
                self.logger.info(f"Successfully added dataset {dataset_name}")
            except Exception as e:
                self.logger.error(f"Failed to add dataset: {str(e)}")
                raise
        else:
            try:
                data_dir = self.datasets_root + data_dir
                class_names = None
                if os.path.exists(data_dir):
                    class_names = os.listdir(data_dir)
                else:
                    self.logger.error(f"Data directory does not exist: {data_dir}")
                self.datasets.append(
                    DatasetInfo(dataset_name, None, data_dir, num_classes, None)
                )
                self.logger.info(f"Successfully added dataset {dataset_name}")
            except Exception as e:
                self.logger.error(f"Failed to add dataset: {str(e)}")
                raise

    def list_datasets(self):
        self.logger.info("Listing all datasets")
        all_datasets = []
        for dataset in self.datasets:
            all_datasets.append(dataset)
            self.logger.info(dataset)
        return all_datasets

    def get_dataset_info(self, dataset_name, domain_name):
        self.logger.info(f"Getting dataset: {dataset_name}, Domain: {domain_name}")

        for dataset in self.datasets:
            if (
                dataset.dataset_name == dataset_name
                and dataset.domain_name == domain_name
            ):
                self.logger.info(
                    f"Dataset found: {dataset_name}, Domain: {domain_name}"
                )
                return dataset
        self.logger.warning(f"Dataset not found: {dataset_name}, Domain: {domain_name}")

        return None
