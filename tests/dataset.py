"""
读取配置文件
"""
import yaml

with

"""
读取数据集
"""

from clip_tuner import load_dataset

dataset = load_dataset(
    datasets_config="default", dataset_name="domainnet", domain_name="real"
)
print(dataset)


from clip_tuner import get_train_transforms, get_val_transforms

train_transforms = get_train_transforms()
