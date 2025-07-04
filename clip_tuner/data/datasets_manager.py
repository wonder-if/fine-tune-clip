import os
import json
import logging

from .utils import make_dataset_dict
from datasets import (
    Dataset,
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


class DatasetManager:
    def __init__(self, datasets_meta=config_file_path, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self.logger.info("DataManager initialized")
        self.datasets = []

        if datasets_meta is not None:
            self.load_from_json(datasets_meta)
        self.list_datasets()

    def load_from_json(self, file_name):
        self.logger.info(f"Loading datasets from JSON file: {file_name}")
        try:
            with open(file_name, "r") as f:
                self.datasets_meta = json.load(f)
            self.datasets_root = self.datasets_meta["root"]
            for dataset_info in self.datasets_meta["datasets"]:
                self.logger.debug(f"Processing dataset: {dataset_info['name']}")
                for domain_info in dataset_info["domains"]:
                    self.add_dataset(
                        dataset_info["name"],
                        dataset_info["dir"],
                        dataset_info["num_classes"],
                        domain_info,
                    )
            self.logger.info("Datasets information loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading datasets from JSON file: {e}")

    # 添加数据集
    def add_dataset(self, dataset_name, data_dir, num_classes, domain_info):
        self.logger.info(
            f"Adding dataset: {dataset_name}, Domain: {domain_info['domain_name']}"
        )

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

    def load_hf_dataset(
        self,
        dataset_name: str,
        domain_name: str,
        split_name: str = "train",
        cache_root: str = "./.cache",
    ):
        """
        1) 尝试从磁盘 cache 秒级加载
        2) 否则，用 ImageFolder+from_dict 构造 DatasetDict
        3) 显式 cast schema (Image + ClassLabel)
        4) 存到 cache 里，返回
        """
        # --- 0) cache 路径 ---
        cache_path = os.path.join(cache_root, dataset_name, domain_name)
        if os.path.isdir(cache_path):
            # 已经有 cache，直接重载
            return load_from_disk(cache_path)

        # --- 1) 把文件夹结构转成 DatasetDict ---
        dataset_path = self.get_dataset_info(dataset_name, domain_name).data_dir
        # make_dataset_dict 内部用 ImageFolder + Dataset.from_dict
        raw_dict = make_dataset_dict(dataset_path, split_name)
        ds = DatasetDict(raw_dict)

        # --- 2) 构造并应用 schema ---
        # 从子文件夹名里拿到所有类别
        classes = (
            ds[split_name].features["label"].names
            if hasattr(ds[split_name].features["label"], "names")
            else sorted(set(ds[split_name]["label"]))
        )
        features = Features(
            {
                "image": HfImage(),  # 声明这一列是图片，按需懒加载
                "label": ClassLabel(names=classes),
            }
        )
        ds = ds.cast(features)  # 应用 schema，等同于 schema inference

        # （可选）如果想后续直接拿到 torch.Tensor：
        # ds.set_format(type="torch", columns=["image", "label"])

        # --- 3) 持久化到磁盘，下次秒级打开 ---
        os.makedirs(cache_path, exist_ok=True)
        ds.save_to_disk(cache_path)

        return ds
