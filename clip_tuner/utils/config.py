from yacs.config import CfgNode as CN
import json

# 定义默认参数
cfg = CN()

# 模型相关参数
cfg.model = CN()
cfg.model.name = "resnet50"
cfg.model.pretrained = True
cfg.model.num_labels = 10

# 数据处理相关参数
cfg.data = CN()
cfg.data.image_size = 224
cfg.data.image_mean = [0.485, 0.456, 0.406]
cfg.data.image_std = [0.229, 0.224, 0.225]
cfg.data.blur = False

# 训练相关参数
cfg.training = CN()
cfg.training.batch_size = 32
cfg.training.lr = 1e-3
cfg.training.num_epochs = 10
cfg.training.optimizer = "sgd"
cfg.training.lr_scheduler = "cosine"


def get_cfg_defaults():
    """获取默认配置"""
    return cfg.clone()


def load_json_config(json_file):
    with open(json_file, "r") as f:
        json_cfg = json.load(f)
    return json_cfg


def merge_from_file(cfg, cfg_file):
    """从文件中加载配置"""
    json_cfg = load_json_config(cfg_file)
    print(json_cfg)
    cfg.merge_from_other_cfg(CN(json_cfg))
