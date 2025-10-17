# clip_tuner 库使用指南

`clip_tuner` 是围绕 HuggingFace CLIP 模型的轻量适配层，提供零样本评估、提示词管理与自定义训练循环。本文面向准备将其作为独立库使用的开发者，覆盖核心 API、配置体系与完整示例。

## 安装与环境准备
- 安装依赖：`pip install -e .`（根目录含 `setup.py` 和 `clip_tuner.egg-info`，支持可编辑安装）。
- 依赖要点：需要 PyTorch、`transformers`, `datasets`, `evaluate`, `hydra-core`, `torchvision` 等；详见项目 `requirements` 或环境配置。
- 推荐关闭 tokenizer 的多线程警告：`os.environ["TOKENIZERS_PARALLELISM"] = "false"`。

## 快速上手示例
以下脚本演示在 Python 环境中直接调用库函数完成模型加载、数据集构建与一次评估：

```python
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from clip_tuner import (
    load_model,
    build_train_eval_dataset,
    build_zero_shot_dataset,
    BaseTrainer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base=None, config_path="pkg://clip_tuner/configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 加载预训练 CLIP 模型及处理器
    model, tokenizer, processor = load_model(
        **cfg.models_info,
        **cfg.pretrained_model,
    )

    # 2. 构造 train/eval/zero-shot 数据集与 collator
    train_dataset, eval_dataset, collator = build_train_eval_dataset(
        cfg.datasets_info,
        cfg.dataset,
        cfg.transforms,
        cfg.prompts,
        processor,
        tokenizer,
    )
    zero_shot_dataset, zero_shot_collator = build_zero_shot_dataset(
        cfg.datasets_info,
        cfg.dataset,
        processor,
    )

    # 3. 创建 trainer 并评估
    trainer = instantiate(
        cfg.trainer,
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        zero_shot_dataset=zero_shot_dataset,
        zero_shot_collator=zero_shot_collator,
        clip_tokenizer=tokenizer,
    )

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
```

使用 Hydra 时，可在命令行覆盖配置，如：`python examples/run.py dataset.train.domain_name=real`.

## Hydra 配置概览
- 入口 `clip_tuner/configs/config.yaml` 通过 defaults 链接各子配置。
- `data/datasets_info/*.yaml`：注册所有数据集的根路径、域划分与类别数，实例化 `_target_: clip_tuner.data.DatasetManager`。
- `data/dataset/*.yaml`：选择当前实验的 train/eval/zero-shot 域组合。
- `data/transforms/*.yaml`：定义 train/eval 的 torchvision transform pipeline；`build_image_transforms` 会用模型 `processor` 的尺寸均值标准差覆盖默认值。
- `data/prompts/*.yaml`：统一管理提示模版字符串（例如 `"a photo of a {}"`）。
- `model/models_info/*.yaml`：列出可用模型及本地缓存路径。
- `model/pretrained_model/*.yaml`：指定当前使用的模型名称。
- `trainer/default.yaml`：声明 `_target_` 为 `clip_tuner.BaseTrainer`，并引用 `trainer/args` 和 `trainer/compute_loss_func`。

## 对外 API 详解

### `load_model(models_root, models_meta, model_name)`
- **功能**：加载指定名称的预训练 CLIP 模型，并返回 `(CLIPModel, CLIPTokenizer, CLIPProcessor)`。
- **参数来源**：通常由 Hydra 配置传入：
  ```yaml
  models_root: /mnt/local-data/workspace/models/
  models_meta:
    - name: clip-vit-base-patch16
      path: modelscope/hub/openai-mirror/clip-vit-base-patch16/
      config: ""
  ```
- **使用示例**：
  ```python
  model, tokenizer, processor = load_model(
      models_root="/path/to/root/",
      models_meta=[{"name": "clip-vit-base-patch16", "path": "openai/clip-vit-base-patch16/", "config": ""}],
      model_name="clip-vit-base-patch16",
  )
  ```
- **注意事项**：函数会通过 `CLIPModel.from_pretrained` 等方法读取本地或远程路径，需要确保路径可访问。

### `build_train_eval_dataset(dataset_info_cfg, dataset_cfg, transforms_cfg, prompts_cfg, processor, tokenizer)`
- **功能**：构建训练集、验证集，并返回文本-图像混合的 collator。
- **流程**：
  1. 基于 `dataset_info_cfg` 实例化 `DatasetManager`，解析数据集目录。
  2. 按 `dataset_cfg.train/eval` 通过 `load_dataset` 拉取对应域的数据（返回 HuggingFace `Dataset`）。
  3. 通过 `build_pixel_prompt_mapping` 将类别名称应用到 `prompts_cfg.prompt_template` 并用 CLIP tokenizer 编码。
  4. 结合模型 `processor` 的参数构建图像变换流水线，写回 `pixel_values`。
  5. 返回 `(train_dataset, eval_dataset, train_collator)`，其中 `train_collator` 负责输出张量化的 `pixel_values/input_ids/attention_mask`。
- **简易调用**：
  ```python
  train_ds, eval_ds, collator = build_train_eval_dataset(
      datasets_info_cfg=cfg.datasets_info,
      dataset_cfg=cfg.dataset,
      transforms_cfg=cfg.transforms,
      prompts_cfg=cfg.prompts,
      processor=processor,
      tokenizer=tokenizer,
  )
  ```
- **拓展**：如需额外字段，可在返回的 `Dataset` 上调用 `.add_column` 或 `.map` 添加自定义处理。

### `build_zero_shot_dataset(dataset_info_cfg, dataset_cfg, processor, cache_path="./.cache")`
- **功能**：针对指定域构建仅包含图像和标签的 zero-shot 数据集与 collator。
- **实现要点**：
  - 内部使用与训练集相同的 `DatasetManager` 和 `load_dataset`。
  - 用 `processor.image_processor` 对图像批处理，输出 `pixel_values`。
  - 支持指定 `cache_path` 将处理后的结果保存在 `.arrow` 文件中，加速重复实验。
- **返回**：`(dataset, zero_shot_collator)`；collator 会把 `pixel_values` 与标签堆叠为张量。
- **示例**：
  ```python
  zs_dataset, zs_collator = build_zero_shot_dataset(
      cfg.datasets_info,
      cfg.dataset,
      processor,
      cache_path="./.cache",
  )
  ```

### `BaseTrainer`
- **继承关系**：扩展自 `transformers.Trainer`。
- **新增能力**：
  - `zero_shot_evaluate`：接受 zero-shot 数据集与 collator，自动生成类别文本输入，并在 GPU/分布式环境保持一致的 dataloader 配置。
  - `get_class_text_inputs`：从数据集 `features["label"].names` 中获取类别名称，并套用 `prompt_template`。
  - `zero_shot_loop`：执行特征提取、计算对称 CLIP logits，并输出整体与按类准确率（若启用）。
- **常见使用**：
  ```python
  trainer = BaseTrainer(
      model=model,
      args=training_args,               # transformers.TrainingArguments
      data_collator=collator,
      train_dataset=train_ds,
      eval_dataset=eval_ds,
      zero_shot_dataset=zs_dataset,
      zero_shot_collator=zs_collator,
      clip_tokenizer=tokenizer,
      prompt_template="a photo of a {}",
      compute_metrics=lambda eval_pred: {"accuracy": ...},  # 可选
  )
  trainer.train()
  zero_shot_metrics = trainer.zero_shot_evaluate(zero_shot_dataset=zs_dataset)
  ```
- **与 Hydra 结合**：在配置中声明 `_target_: clip_tuner.BaseTrainer`，并注入 `trainer.args` 作为 `TrainingArguments`，即可用 `instantiate(cfg.trainer, ...)` 生成实例。

### `SymmetricCLIPLoss(temperature: float = 0.07)`
- **定位**：默认的训练损失，用于对称地计算图像到文本、文本到图像的交叉熵。
- **使用方式**：在 Hydra 配置 `trainer/compute_loss_func/default.yaml` 中指定 `_target_: clip_tuner.SymmetricCLIPLoss`。
- **自定义**：若需替换损失函数，可在配置中指向新的 `_target_` 并匹配 `Trainer` 的 `compute_loss` 签名。

## 目录结构速览
- `clip_tuner/data/`：`DatasetManager`、图像变换、prompt tokenizer，支撑 `build_*` API。
- `clip_tuner/models/`：`ModelManager` 和可学习提示嵌入 `CLIPTextLearnableEmbeddings`。
- `clip_tuner/trainers/`：`base_trainer.py` 中包含 zero-shot 评估逻辑；`utils.py` 提供 accuracy 计算；`accuracy/` 子目录打包了 evaluate 插件。
- `clip_tuner/losses/clip_loss.py`：对称对比损失实现。
- `clip_tuner/utils/`：配置、日志、可视化等通用工具。
- `clip_tuner/__init__.py`：集中导出本文档介绍的所有 API。

## 周边工具
- `exps/exp_1_zero_shot_baseline.py`：基于 Hydra 的参考脚本，演示仅以 zero-shot 数据驱动的评估流程。
- `tests/dataset.py`：端到端地加载模型、构建数据集，并运行一次 `trainer.evaluate()`，适合作为安装后快速验收。
- `tests/prompt_domain_experiment.py`：提供跨域 prompt 分析、特征提取与统计检验的高级脚本。
- `scripts/exp_1_zero_shot.sh`：Shell 示例，循环不同域并解析 JSON 输出写入 CSV。
- `docs/`：附加设计文档与调研记录。

## 常见扩展路径
- 新增数据集：在 `configs/data/datasets_info` 注册根目录与域，再在 `configs/data/dataset` 创建组合配置。
- 切换模型：在 `configs/model/models_info` 添加条目，通过 `model/pretrained_model` 指定默认使用的模型名。
- 引入可学习 prompt：调用 `clip_tuner.models.add_learnable_prompts_to_clip_text_model` 将 learnable token 注入现有模型。
- 自定义度量：实现返回字典的函数，传入 `BaseTrainer(..., compute_metrics=my_fn)` 即可。

通过以上 API，你可以在不改动内部实现的情况下，快速完成 CLIP 的微调、零样本评估或 prompt 实验。欢迎结合 Hydra 的配置继承与命令行覆盖能力，构建更复杂的实验流程。
