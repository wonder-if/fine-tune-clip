from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def compute_image_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> dict:
    """提取整批图像特征，并返回 features/labels/metadata。"""
    model.eval()
    device = device or next(model.parameters()).device

    feats_list = []
    labels_list = []
    seen = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding images", leave=False)

    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values)
            if normalize:
                feats = F.normalize(feats, dim=-1)

        feats_list.append(feats.detach().cpu().to(dtype=dtype))
        labels_list.append(labels.detach().cpu())

        seen += labels.size(0)
        if max_samples is not None and seen >= max_samples:
            break

    features = torch.cat(feats_list, dim=0)
    label_tensor = torch.cat(labels_list, dim=0)

    if max_samples is not None and features.size(0) > max_samples:
        features = features[:max_samples]
        label_tensor = label_tensor[:max_samples]

    metadata = {
        "num_samples": int(features.size(0)),
        "feature_dim": int(features.size(-1)),
        "labels_provided": True,
    }

    ds = getattr(dataloader, "dataset", None)
    label_feature = getattr(getattr(ds, "features", None), "get", lambda *_: None)(
        "label"
    )
    if label_feature is None and hasattr(ds, "features"):
        label_feature = ds.features.get("label")
    if label_feature is not None:
        names = getattr(label_feature, "names", None)
        if names:
            metadata["class_names"] = list(names)

    return {
        "features": features,
        "labels": label_tensor,
        "metadata": metadata,
    }


def compute_text_features(
    model: torch.nn.Module,
    tokenizer,
    prompts: Iterable[str],
    device: Optional[torch.device] = None,
    *,
    normalize: bool = True,
) -> dict:
    """基于提示词列表提取文本特征。"""
    model.eval()
    device = device or next(model.parameters()).device

    prompts = list(prompts)
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**encoded)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)

    return {
        "features": text_features.detach().cpu(),
        "metadata": {
            "num_prompts": len(prompts),
            "feature_dim": text_features.size(-1),
            "normalize": normalize,
        },
    }


class FeatureTensorDataset(Dataset):
    """简单的张量数据集，既可返回图像特征，也能按类别索引文本特征。"""

    def __init__(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
        text_features: Optional[torch.Tensor] = None,
        prompts: Optional[Iterable[str]] = None,
    ):
        if features.dim() != 2:
            raise ValueError("features tensor must be [N, D]")
        self.features = features.clone().detach()
        if labels is not None and not torch.is_tensor(labels):
            labels = torch.tensor(labels)
        self.labels = (
            labels.clone().detach()
            if labels is not None
            else torch.empty(0, dtype=torch.long)
        )
        self.has_labels = labels is not None and labels.numel() == features.size(0)
        self.metadata = metadata or {}
        self.class_names = self.metadata.get("class_names")
        self.prompts = list(prompts) if prompts is not None else None

        if text_features is not None:
            if text_features.dim() != 2:
                raise ValueError("text_features tensor must be [C, D]")
            self.text_features = text_features.clone().detach()
        else:
            self.text_features = None

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> dict:
        item = {"features": self.features[index]}
        if self.has_labels:
            label = self.labels[index]
            item["labels"] = label
            if self.text_features is not None:
                label_idx = int(label.item()) if torch.is_tensor(label) else int(label)
                item["text_features"] = self.text_features[label_idx]
        if self.prompts is not None:
            item["prompts"] = self.prompts
        return item


def feature_collator(examples: Iterable[dict]) -> dict:
    """把 FeatureTensorDataset 的样本拼成 batch。"""
    features = torch.stack([ex["features"] for ex in examples])
    batch = {"features": features}
    if "labels" in examples[0]:
        labels = [ex["labels"] for ex in examples]
        batch["labels"] = (
            torch.stack(labels) if torch.is_tensor(labels[0]) else torch.tensor(labels)
        )
    else:
        batch["labels"] = None
    if "text_features" in examples[0]:
        texts = [ex["text_features"] for ex in examples]
        batch["text_features"] = (
            torch.stack(texts) if torch.is_tensor(texts[0]) else torch.tensor(texts)
        )
    if "prompts" in examples[0]:
        batch["prompts"] = examples[0]["prompts"]
    return batch


def build_feature_dataset(
    payload: dict,
    *,
    text_payload: Optional[dict] = None,
) -> FeatureTensorDataset:
    """将缓存结果包装成 FeatureTensorDataset，可附带文本特征与 prompts。"""
    text_features = None
    prompts = None
    if text_payload is not None:
        text_features = text_payload.get("features")
        meta = text_payload.get("metadata", {})
        prompts = meta.get("prompts") or meta.get("class_names")

    return FeatureTensorDataset(
        features=payload["features"],
        labels=payload.get("labels"),
        metadata=payload.get("metadata"),
        text_features=text_features,
        prompts=prompts,
    )


def build_cached_feature_dataset(
    *,
    model,
    tokenizer,
    datasets_info_cfg,
    dataset_cfg,
    processor,
    model_name: str,
    cache_root: str = "./.cache/features",
    device=None,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    max_samples: Optional[int] = None,
    show_progress: bool = False,
    include_text: bool = True,
    prompt_template: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[FeatureTensorDataset, Callable]:
    """根据配置构建缓存特征数据集，缓存 miss 时会自动推理并写入磁盘。"""
    from clip_tuner.data.zero_shot import build_zero_shot_dataset
    from .cache import cache_image_features, cache_text_features

    zero_shot_dataset, zero_shot_collator = build_zero_shot_dataset(
        datasets_info_cfg,
        dataset_cfg,
        processor,
    )

    effective_device = device or next(model.parameters()).device

    dataloader = DataLoader(
        zero_shot_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=zero_shot_collator,
    )

    zero_shot_cfg = dataset_cfg.zero_shot
    dataset_name = zero_shot_cfg.dataset_name
    domain_name = getattr(zero_shot_cfg, "domain_name", None)
    split_name = getattr(zero_shot_cfg, "split", "zero_shot")

    image_payload = cache_image_features(
        model,
        dataloader,
        cache_root=cache_root,
        dataset_name=dataset_name,
        domain_name=domain_name,
        split=split_name,
        model_name=model_name,
        device=effective_device,
        normalize=normalize,
        dtype=dtype,
        max_samples=max_samples,
        show_progress=show_progress,
        force_refresh=force_refresh,
    )

    text_payload = None
    if include_text:
        if tokenizer is None:
            raise ValueError("Tokenizer is required when include_text=True.")
        prompt_template = prompt_template or "a photo of a {}"
        class_names = image_payload["metadata"].get("class_names")
        if (
            not class_names
            and hasattr(zero_shot_dataset, "features")
            and "label" in zero_shot_dataset.features
        ):
            label_feature = zero_shot_dataset.features["label"]
            names = getattr(label_feature, "names", None)
            if names:
                class_names = names
        if not class_names:
            raise ValueError("Unable to resolve class names for text feature caching.")

        text_payload = cache_text_features(
            model,
            tokenizer,
            cache_root=cache_root,
            dataset_name=dataset_name,
            domain_name=domain_name,
            split=split_name,
            model_name=model_name,
            prompt_template=prompt_template,
            class_names=class_names,
            device=effective_device,
            normalize=normalize,
            force_refresh=force_refresh,
        )

    feature_dataset = build_feature_dataset(
        image_payload,
        text_payload=text_payload,
    )

    return feature_dataset, feature_collator
