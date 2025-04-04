import os
import json
import logging

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
        return f"DatasetInfo(dataset_name={self.dataset_name}, domain_name={self.domain_name}, data_dir={self.data_dir}, num_classes={self.num_classes}, class_names={self.class_names})"


class DataManager:
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
            self.logger.info(f"Successfully added dataset{dataset_name}")
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

    def get_dataset(self, dataset_name, domain_name):
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
