from data import load_dataset
from data import Features, Value
from rich import print

dataset_root = r"/mnt/nas-data/work-data/datasets/images/"

dataset_name = "office-31"
domain_name = "amazon"
# 加载数据集
dataset = load_dataset(
    "imagefolder",
    data_dir=rf"D:\DLSources\Datasets\Images\DomainAdaptation\{dataset_name}\{domain_name}",
    split="train",
)
