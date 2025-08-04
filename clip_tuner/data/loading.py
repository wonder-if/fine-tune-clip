import os
import logging

from torchvision.datasets import ImageFolder

from datasets import (
    Dataset,
    load_dataset as load_hf_dataset,
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
        self.logger.debug("DataManager initialized")
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
            self.logger.debug("Datasets information loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading datasets from JSON file: {e}")
            raise

    # 添加数据集
    def add_dataset(self, dataset_name, data_dir, num_classes, domain_info):
        self.logger.debug(
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
                self.logger.debug(f"Successfully added dataset {dataset_name}")
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
                self.logger.debug(f"Successfully added dataset {dataset_name}")
            except Exception as e:
                self.logger.error(f"Failed to add dataset: {str(e)}")
                raise

    def list_datasets(self):
        self.logger.debug("Listing all datasets")
        all_datasets = []
        for dataset in self.datasets:
            all_datasets.append(dataset)
            self.logger.info(dataset)
        return all_datasets

    def get_dataset_info(self, dataset_name, domain_name):
        self.logger.debug(f"Getting dataset: {dataset_name}, Domain: {domain_name}")

        for dataset in self.datasets:
            if (
                dataset.dataset_name == dataset_name
                and dataset.domain_name == domain_name
            ):
                self.logger.debug(
                    f"Dataset found: {dataset_name}, Domain: {domain_name}"
                )
                return dataset
        self.logger.warning(f"Dataset not found: {dataset_name}, Domain: {domain_name}")

        return None


# def make_dataset_dict(path: str, split_name: str = "train"):
#     """
#     Load images from a folder path using torchvision ImageFolder,
#     convert to a HuggingFace Dataset, and return a dict {split_name: dataset}.

#     Args:
#         path (str): Path to the folder containing class subfolders.
#         split_name (str): Name to assign to the resulting split key.

#     Returns:
#         dict: A mapping from split_name to a HuggingFace Dataset.

#     >>> dataset_dict = make_dataset_dict("path/to/folder")
#     >>> from datasets import DatasetDict
#     >>> dataset_dict = DatasetDict(datasets_dict)
#     >>> dataset_dict

#     """
#     # Use ImageFolder to scan subfolders as classes
#     torch_ds = ImageFolder(root=path)

#     # Extract image file paths and corresponding labels
#     image_paths, labels = zip(*torch_ds.samples)

#     # Define HF Dataset features: Image + ClassLabel
#     features = Features(
#         {"image": HfImage(), "label": ClassLabel(names=torch_ds.classes)}
#     )

#     # Create a HF Dataset from the collected paths and labels
#     hf_ds = Dataset.from_dict(
#         {"image": list(image_paths), "label": list(labels)}, features=features
#     )

#     return {split_name: hf_ds}


# def load_hf_dataset_from_dir(
#     dataset_path: str,
#     cache_path: str,
#     split_name: str = "train",
# ):
#     """
#     1) 尝试从磁盘 cache 秒级加载
#     2) 否则，用 ImageFolder+from_dict 构造 DatasetDict
#     3) 显式 cast schema (Image + ClassLabel)
#     4) 存到 cache 里，返回
#     """
#     # --- 0) cache 路径 ---
#     if os.path.isdir(cache_path):
#         # 已经有 cache，直接重载
#         ds = load_from_disk(cache_path)
#         return ds[split_name]

#     # --- 1) 把文件夹结构转成 DatasetDict ---
#     # make_dataset_dict 内部用 ImageFolder + Dataset.from_dict
#     raw_dict = make_dataset_dict(dataset_path, split_name)
#     ds = DatasetDict(raw_dict)

#     # --- 2) 构造并应用 schema ---
#     # 从子文件夹名里拿到所有类别
#     classes = (
#         ds[split_name].features["label"].names
#         if hasattr(ds[split_name].features["label"], "names")
#         else sorted(set(ds[split_name]["label"]))
#     )
#     features = Features(
#         {
#             "image": HfImage(),  # 声明这一列是图片，按需懒加载
#             "label": ClassLabel(names=classes),
#         }
#     )
#     ds = ds.cast(features)  # 应用 schema，等同于 schema inference

#     # --- 3) 持久化到磁盘，下次秒级打开 ---
#     os.makedirs(cache_path, exist_ok=True)
#     ds.save_to_disk(cache_path)

#     return ds[split_name]


def load_dataset(
    data_manager: DatasetManager = None,
    dataset_name: str = None,
    domain_name: str = None,
    cache_root: str = "./.cache",
) -> Dataset:
    dataset_info = data_manager.get_dataset_info(dataset_name, domain_name)
    cache_path = os.path.join(cache_root, dataset_name)
    if domain_name:
        cache_path = os.path.join(cache_path, domain_name)
    dataset = load_hf_dataset(
        dataset_info.data_dir,
        cache_dir=cache_path,
    )
    # print(dataset)
    return dataset["train"]
