# Feature Cache Utilities

When analysing multiple domains or prompts it is convenient to reuse frozen model features instead of recomputing them every run. The helpers in `clip_tuner.features` provide a thin cache layer for both the image and text encoders.

## Quick start

```python
from torch.utils.data import DataLoader
from clip_tuner import build_cached_feature_dataset

feature_dataset, feature_collator = build_cached_feature_dataset(
    model=model,
    tokenizer=tokenizer,
    datasets_info_cfg=cfg.datasets_info,
    dataset_cfg=cfg.dataset,
    processor=processor,
    cache_root="./.cache/features",
    model_name=cfg.pretrained_model.model_name,
    include_text=True,
    prompt_template="a photo of a {}",
)

feature_loader = DataLoader(
    feature_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=feature_collator,
)
# 每个 batch 现在同时包含 image_features / labels / text_features（可选）/ prompts
```

- The first call writes the payload to `./.cache/features/...`. Subsequent calls reuse the file unless `force_refresh=True` is passed.
- Cached tensors are stored on CPU (`float32` by default) together with metadata (`num_samples`, `class_names`, cache parameters).
- Paths are derived from dataset/domain/split/model; prompt-dependent text caches include a hash of the template to avoid collisions.

These helpers are intentionally low level so you can plug them into any experiment script inside `exps/` without rewriting extraction logic.

## Using cache vs. raw images transparently

- 首次调用会触发模型前向并写入缓存，后续命中直接加载；需要重算时传 `force_refresh=True`。
- 若仍需原始图像流程（例如数据增强），照旧使用 HuggingFace Dataset + collator 即可。
- 想改成缓存特征，只需调用 `build_cached_feature_dataset` 获取 `FeatureTensorDataset` 与 `feature_collator`，后续训练/评估代码无需改动。
