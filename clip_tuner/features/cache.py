import hashlib
import os
from typing import Iterable, Optional

import torch

from .extract import compute_image_features, compute_text_features


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _prompt_hash(prompt_template: str) -> str:
    return hashlib.sha1(prompt_template.encode("utf-8")).hexdigest()[:16]


def image_cache_path(
    cache_root: str,
    dataset_name: str,
    domain_name: Optional[str],
    split: Optional[str],
    model_name: str,
) -> str:
    parts = [
        cache_root,
        "image",
        dataset_name,
        domain_name or "default",
        split or "unspecified",
        f"{model_name}.pt",
    ]
    return os.path.join(*parts)


def text_cache_path(
    cache_root: str,
    dataset_name: str,
    domain_name: Optional[str],
    split: Optional[str],
    model_name: str,
    prompt_template: str,
) -> str:
    prompt_hash = _prompt_hash(prompt_template)
    parts = [
        cache_root,
        "text",
        dataset_name,
        domain_name or "default",
        split or "unspecified",
        f"{model_name}-{prompt_hash}.pt",
    ]
    return os.path.join(*parts)


def cache_image_features(
    model,
    dataloader,
    *,
    cache_root: str = "./.cache/features",
    dataset_name: str,
    domain_name: Optional[str],
    split: Optional[str],
    model_name: str,
    device=None,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    max_samples: Optional[int] = None,
    show_progress: bool = False,
    force_refresh: bool = False,
) -> dict:
    path = image_cache_path(
        cache_root, dataset_name, domain_name, split, model_name
    )

    if not force_refresh and os.path.exists(path):
        return torch.load(path, map_location="cpu")

    payload = compute_image_features(
        model,
        dataloader,
        device=device,
        normalize=normalize,
        dtype=dtype,
        max_samples=max_samples,
        show_progress=show_progress,
    )
    payload.setdefault("metadata", {})
    payload["metadata"].update(
        {
            "dataset_name": dataset_name,
            "domain_name": domain_name,
            "split": split,
            "model_name": model_name,
            "normalize": normalize,
            "dtype": str(dtype),
            "max_samples": max_samples,
        }
    )

    _ensure_parent(path)
    torch.save(payload, path)
    return payload


def cache_text_features(
    model,
    tokenizer,
    *,
    cache_root: str = "./.cache/features",
    dataset_name: str,
    domain_name: Optional[str],
    split: Optional[str],
    model_name: str,
    prompt_template: str,
    class_names: Iterable[str],
    device=None,
    normalize: bool = True,
    force_refresh: bool = False,
) -> dict:
    path = text_cache_path(
        cache_root, dataset_name, domain_name, split, model_name, prompt_template
    )

    if not force_refresh and os.path.exists(path):
        return torch.load(path, map_location="cpu")

    prompts = [prompt_template.format(name) for name in class_names]
    payload = compute_text_features(
        model,
        tokenizer,
        prompts,
        device=device,
        normalize=normalize,
    )
    payload.setdefault("metadata", {})
    payload["metadata"].update(
        {
            "dataset_name": dataset_name,
            "domain_name": domain_name,
            "split": split,
            "model_name": model_name,
            "prompt_template": prompt_template,
            "normalize": normalize,
            "class_names": list(class_names),
            "prompts": prompts,
        }
    )

    _ensure_parent(path)
    torch.save(payload, path)
    return payload


__all__ = [
    "cache_image_features",
    "cache_text_features",
    "image_cache_path",
    "text_cache_path",
]
