import os

from torchvision.datasets import ImageFolder

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    ClassLabel,
    Image as HfImage,
    load_from_disk,
)

from .dataset_manager import DatasetManager


def make_dataset_dict(path: str, split_name: str = "train"):
    """
    Load images from a folder path using torchvision ImageFolder,
    convert to a HuggingFace Dataset, and return a dict {split_name: dataset}.

    Args:
        path (str): Path to the folder containing class subfolders.
        split_name (str): Name to assign to the resulting split key.

    Returns:
        dict: A mapping from split_name to a HuggingFace Dataset.

    >>> dataset_dict = make_dataset_dict("path/to/folder")
    >>> from datasets import DatasetDict
    >>> dataset_dict = DatasetDict(datasets_dict)
    >>> dataset_dict

    """
    # Use ImageFolder to scan subfolders as classes
    torch_ds = ImageFolder(root=path)

    # Extract image file paths and corresponding labels
    image_paths, labels = zip(*torch_ds.samples)

    # Define HF Dataset features: Image + ClassLabel
    features = Features(
        {"image": HfImage(), "label": ClassLabel(names=torch_ds.classes)}
    )

    # Create a HF Dataset from the collected paths and labels
    hf_ds = Dataset.from_dict(
        {"image": list(image_paths), "label": list(labels)}, features=features
    )

    return {split_name: hf_ds}


def load_hf_dataset_from_dir(
    dataset_path: str,
    cache_path: str,
    split_name: str = "train",
):
    """
    1) 尝试从磁盘 cache 秒级加载
    2) 否则，用 ImageFolder+from_dict 构造 DatasetDict
    3) 显式 cast schema (Image + ClassLabel)
    4) 存到 cache 里，返回
    """
    # --- 0) cache 路径 ---
    if os.path.isdir(cache_path):
        # 已经有 cache，直接重载
        ds = load_from_disk(cache_path)
        return ds[split_name]

    # --- 1) 把文件夹结构转成 DatasetDict ---
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

    return ds[split_name]


def load_dataset(
    datasets_root: str = None,
    datasets_meta=None,
    dataset_name: str = None,
    domain_name: str = None,
    cache_root: str = "./.cache",
) -> Dataset:

    data_manager = DatasetManager(datasets_root, datasets_meta)

    dataset_info = data_manager.get_dataset_info(dataset_name, domain_name)
    cache_path = os.path.join(cache_root, dataset_name)
    if domain_name:
        cache_path = os.path.join(cache_path, domain_name)
    return load_hf_dataset_from_dir(
        dataset_info.data_dir,
        cache_path=cache_path,
    )
