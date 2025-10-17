import json
import logging
import math
import os
import statistics
import warnings
from typing import Dict, Iterable, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip_tuner import build_zero_shot_dataset, load_model


LOGGER = logging.getLogger(__name__)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def set_seed(seed: int) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def prepare_domain_dataset(
    datasets_info_cfg: DictConfig,
    domain_cfg: DictConfig,
    processor,
    cache_root: str,
) -> Tuple:
    domain_dataset_cfg = OmegaConf.create(
        {
            "zero_shot": {
                "dataset_name": domain_cfg.dataset_name,
                "domain_name": domain_cfg.domain_name,
                "cache_root": cache_root,
            }
        }
    )

    dataset, collator = build_zero_shot_dataset(
        datasets_info_cfg,
        domain_dataset_cfg,
        processor,
        cache_path=cache_root,
    )

    if domain_cfg.max_samples is not None:
        take = min(domain_cfg.max_samples, len(dataset))
        dataset = dataset.select(range(take))

    return dataset, collator


def make_dataloader(
    dataset,
    collator,
    eval_cfg: DictConfig,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=eval_cfg.pin_memory,
        persistent_workers=eval_cfg.persistent_workers,
        prefetch_factor=eval_cfg.prefetch_factor,
        collate_fn=collator,
    )
    return loader


def gather_domain_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    collected = 0

    for batch in tqdm(dataloader, desc="Extracting image features", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        batch_labels = batch["labels"].to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

        features.append(image_features.detach().to(dtype=torch.float32).cpu())
        labels.append(batch_labels.detach().cpu())

        collected += batch_labels.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    if max_samples is not None and features_tensor.size(0) > max_samples:
        features_tensor = features_tensor[:max_samples]
        labels_tensor = labels_tensor[:max_samples]

    if features_tensor.dtype != dtype:
        features_tensor = features_tensor.to(dtype)

    return features_tensor, labels_tensor


def chunked_logits(
    features: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    logits: List[torch.Tensor] = []
    for start in range(0, features.size(0), chunk_size):
        end = min(start + chunk_size, features.size(0))
        feature_chunk = features[start:end].to(device)
        with torch.no_grad():
            logits_chunk = feature_chunk @ text_features.T
        logits.append(logits_chunk.cpu())
    return torch.cat(logits, dim=0)


def evaluate_prompt(
    features: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    topk: Iterable[int],
) -> Tuple[Dict[str, float], List[int]]:
    logits = chunked_logits(features, text_features, device, chunk_size)
    labels_cpu = labels.cpu()
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels_cpu).to(torch.int)

    metrics: Dict[str, float] = {}
    num_samples = labels_cpu.size(0)

    topk_unique = sorted(set(int(k) for k in topk if k >= 1))
    for k in topk_unique:
        effective_k = min(k, logits.size(1))
        topk_hits = logits.topk(k=effective_k, dim=-1).indices == labels_cpu.unsqueeze(-1)
        topk_correct = topk_hits.any(dim=-1).to(torch.float32)
        metrics[f"top{k}_accuracy"] = topk_correct.mean().item()

    if "top1_accuracy" not in metrics:
        metrics["top1_accuracy"] = correct.float().mean().item()

    metrics["num_samples"] = num_samples
    return metrics, correct.tolist()


def tokenize_prompts(tokenizer, prompt_template: str, class_names: List[str], device: torch.device):
    prompts = [prompt_template.format(name) for name in class_names]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def compute_text_features(
    model: torch.nn.Module,
    tokenizer,
    prompt_template: str,
    class_names: List[str],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    tokenized = tokenize_prompts(tokenizer, prompt_template, class_names, device)
    with torch.no_grad():
        text_features = model.get_text_features(**tokenized)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def compute_covariance(features: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(features.size(0) - 1, 1)
    if eps > 0.0:
        cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return cov


def compute_fid_value(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    eps: float,
) -> float:
    try:
        from scipy import linalg
    except ImportError as exc:
        raise RuntimeError(
            "FID requested but scipy is not available in the environment."
        ) from exc

    mu_a = features_a.mean(dim=0).numpy()
    mu_b = features_b.mean(dim=0).numpy()
    cov_a = compute_covariance(features_a, eps=eps).numpy()
    cov_b = compute_covariance(features_b, eps=eps).numpy()

    diff = mu_a - mu_b

    covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn("FID computation produced non-finite values; adding jitter.")
        cov_a += np.eye(cov_a.shape[0]) * eps
        cov_b += np.eye(cov_b.shape[0]) * eps
        covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(cov_a + cov_b - 2 * covmean)
    return float(fid)


def compute_feature_distances(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    cfg: DictConfig,
) -> Dict[str, float]:
    metrics = {}
    metrics_cfg = cfg.metrics

    mu_a = F.normalize(features_a.mean(dim=0), dim=0)
    mu_b = F.normalize(features_b.mean(dim=0), dim=0)

    if metrics_cfg.cosine_shift:
        cosine = torch.dot(mu_a, mu_b).clamp(-1.0, 1.0).item()
        metrics["cosine_shift"] = 1.0 - cosine

    if metrics_cfg.euclidean:
        metrics["euclidean_distance"] = torch.linalg.norm(
            features_a.mean(dim=0) - features_b.mean(dim=0)
        ).item()

    if metrics_cfg.covariance_fro:
        cov_a = compute_covariance(features_a)
        cov_b = compute_covariance(features_b)
        metrics["covariance_fro_norm"] = torch.linalg.matrix_norm(
            cov_a - cov_b
        ).item()

    if metrics_cfg.fid:
        try:
            metrics["fid"] = compute_fid_value(
                features_a, features_b, cfg.fid_eps
            )
        except RuntimeError as err:
            warnings.warn(str(err))

    return metrics


def compute_prompt_sensitivity(
    accuracy_matrix: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    stats = {}
    for domain_label, accuracies in accuracy_matrix.items():
        if not accuracies:
            continue
        stats[domain_label] = {
            "std": float(statistics.pstdev(accuracies)),
            "range": float(max(accuracies) - min(accuracies)),
        }
    return stats


def compute_domain_sensitivity(
    prompt_order: List[str],
    accuracy_matrix: Dict[str, List[float]],
    domain_labels: List[str],
) -> Dict[str, float]:
    if len(domain_labels) != 2:
        raise ValueError("Domain sensitivity currently expects exactly two domains.")
    domain_a, domain_b = domain_labels
    acc_a = accuracy_matrix[domain_a]
    acc_b = accuracy_matrix[domain_b]
    return {
        prompt: float(abs(acc_a[idx] - acc_b[idx]))
        for idx, prompt in enumerate(prompt_order)
    }


def prepare_anova_data(
    per_sample_correct: Dict[str, Dict[str, List[int]]],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[int]]], int]:
    domain_labels = sorted(per_sample_correct.keys())
    prompt_labels = sorted(
        {prompt for domain in per_sample_correct.values() for prompt in domain.keys()}
    )

    trimmed = {d: {} for d in domain_labels}
    min_count = math.inf
    for domain in domain_labels:
        for prompt in prompt_labels:
            values = per_sample_correct[domain][prompt]
            trimmed[domain][prompt] = list(values)
            min_count = min(min_count, len(values))

    return domain_labels, prompt_labels, trimmed, min_count


def run_two_way_anova(
    per_sample_correct: Dict[str, Dict[str, List[int]]],
    epsilon: float,
    clip_to_min: bool = True,
) -> Dict[str, Dict[str, float]]:
    domain_labels, prompt_labels, data, min_count = prepare_anova_data(
        per_sample_correct
    )

    if clip_to_min and min_count < math.inf:
        for domain in domain_labels:
            for prompt in prompt_labels:
                data[domain][prompt] = data[domain][prompt][:min_count]

    cell_values = []
    for domain in domain_labels:
        for prompt in prompt_labels:
            cell_values.extend(data[domain][prompt])

    if not cell_values:
        raise ValueError("No per-sample correctness data available for ANOVA.")

    grand_mean = statistics.mean(cell_values)

    domain_means = {
        domain: statistics.mean(
            [score for prompt in prompt_labels for score in data[domain][prompt]]
        )
        for domain in domain_labels
    }
    prompt_means = {
        prompt: statistics.mean(
            [score for domain in domain_labels for score in data[domain][prompt]]
        )
        for prompt in prompt_labels
    }
    cell_means = {
        domain: {
            prompt: statistics.mean(data[domain][prompt]) for prompt in prompt_labels
        }
        for domain in domain_labels
    }

    rep_count = len(next(iter(next(iter(data.values())).values())))
    if rep_count == 0:
        raise ValueError("ANOVA requires at least one observation per cell.")
    total_obs = len(cell_values)
    ss_total = sum((value - grand_mean) ** 2 for value in cell_values)

    ss_domain = sum(
        len(prompt_labels) * rep_count * (domain_means[domain] - grand_mean) ** 2
        for domain in domain_labels
    )
    ss_prompt = sum(
        len(domain_labels) * rep_count * (prompt_means[prompt] - grand_mean) ** 2
        for prompt in prompt_labels
    )
    ss_interaction = 0.0
    for domain in domain_labels:
        for prompt in prompt_labels:
            cell_mean = cell_means[domain][prompt]
            ss_interaction += rep_count * (
                cell_mean - domain_means[domain] - prompt_means[prompt] + grand_mean
            ) ** 2
    ss_residual = max(ss_total - ss_domain - ss_prompt - ss_interaction, 0.0)

    df_domain = len(domain_labels) - 1
    df_prompt = len(prompt_labels) - 1
    df_interaction = df_domain * df_prompt
    df_residual = (
        len(domain_labels) * len(prompt_labels) * (rep_count - 1)
    )

    def safe_mean_square(ss: float, df: int) -> float:
        if df <= 0:
            return float("nan")
        return ss / max(df, epsilon)

    ms_domain = safe_mean_square(ss_domain, df_domain)
    ms_prompt = safe_mean_square(ss_prompt, df_prompt)
    ms_interaction = safe_mean_square(ss_interaction, df_interaction)
    ms_residual = safe_mean_square(ss_residual, df_residual)

    def safe_f_stat(ms_factor: float, ms_error: float) -> float:
        if math.isnan(ms_factor) or math.isnan(ms_error) or ms_error == 0.0:
            return float("nan")
        return ms_factor / max(ms_error, epsilon)

    f_domain = safe_f_stat(ms_domain, ms_residual)
    f_prompt = safe_f_stat(ms_prompt, ms_residual)
    f_interaction = safe_f_stat(ms_interaction, ms_residual)

    from torch.distributions import FisherSnedecor

    def f_p_value(f_value: float, df_num: int, df_den: int) -> float:
        if math.isnan(f_value) or df_num <= 0 or df_den <= 0:
            return float("nan")
        dist = FisherSnedecor(df_num, df_den)
        return float(max(0.0, 1.0 - dist.cdf(torch.tensor(f_value)).item()))

    eta_sq_domain = ss_domain / ss_total if ss_total > 0 else float("nan")
    eta_sq_prompt = ss_prompt / ss_total if ss_total > 0 else float("nan")
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else float("nan")

    return {
        "sum_squares": {
            "domain": ss_domain,
            "prompt": ss_prompt,
            "interaction": ss_interaction,
            "residual": ss_residual,
            "total": ss_total,
        },
        "degrees_of_freedom": {
            "domain": df_domain,
            "prompt": df_prompt,
            "interaction": df_interaction,
            "residual": df_residual,
            "total": total_obs - 1,
        },
        "mean_square": {
            "domain": ms_domain,
            "prompt": ms_prompt,
            "interaction": ms_interaction,
            "residual": ms_residual,
        },
        "f_stat": {
            "domain": f_domain,
            "prompt": f_prompt,
            "interaction": f_interaction,
        },
        "p_value": {
            "domain": f_p_value(f_domain, df_domain, df_residual),
            "prompt": f_p_value(f_prompt, df_prompt, df_residual),
            "interaction": f_p_value(f_interaction, df_interaction),
        },
        "eta_squared": {
            "domain": eta_sq_domain,
            "prompt": eta_sq_prompt,
            "interaction": eta_sq_interaction,
        },
    }


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


@hydra.main(
    version_base=None,
    config_path="pkg://clip_tuner/configs",
    config_name="prompt_domain_experiment",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Running prompt-domain sensitivity analysis.")

    set_seed(cfg.get("seed"))

    model, tokenizer, processor = load_model(
        **cfg.models_info,
        **cfg.pretrained_model,
    )
    for param in model.parameters():
        param.requires_grad_(False)

    device = resolve_device(cfg.evaluation.device)
    dtype = getattr(torch, cfg.evaluation.dtype)
    model = model.to(device)
    model.eval()

    domain_features: Dict[str, torch.Tensor] = {}
    domain_labels_tensor: Dict[str, torch.Tensor] = {}
    domain_class_names: Dict[str, List[str]] = {}
    domain_prompt_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_sample_correct: Dict[str, Dict[str, List[int]]] = {}
    accuracy_matrix: Dict[str, List[float]] = {}

    prompt_templates = list(cfg.prompts.templates)
    compute_topk = cfg.evaluation.compute_topk

    reference_class_names: List[str] = []

    for domain_cfg in cfg.domains:
        dataset, collator = prepare_domain_dataset(
            cfg.datasets_info,
            domain_cfg,
            processor,
            cache_root=cfg.evaluation.cache_root,
        )
        dataset_len = len(dataset)

        if cfg.evaluation.max_samples_per_domain is not None:
            take = min(cfg.evaluation.max_samples_per_domain, dataset_len)
            dataset = dataset.select(range(take))

        dataloader = make_dataloader(dataset, collator, cfg.evaluation)

        features, labels = gather_domain_features(
            model,
            dataloader,
            device=device,
            dtype=getattr(torch, cfg.evaluation.dtype),
            max_samples=cfg.features.max_samples,
        )

        domain_key = domain_cfg.get("label") or domain_cfg.domain_name
        domain_features[domain_key] = features
        domain_labels_tensor[domain_key] = labels

        class_names = list(dataset.features["label"].names)
        if not reference_class_names:
            reference_class_names = class_names
        elif reference_class_names != class_names:
            raise ValueError(
                "Label sets mismatch between domains: "
                f"{reference_class_names[:3]}..., {class_names[:3]}..."
            )
        domain_class_names[domain_key] = class_names
        domain_prompt_metrics[domain_key] = {}
        per_sample_correct[domain_key] = {}
        accuracy_matrix[domain_key] = []

        for prompt_template in prompt_templates:
            text_features = compute_text_features(
                model,
                tokenizer,
                prompt_template,
                domain_class_names[domain_key],
                device,
            )
            metrics, correct = evaluate_prompt(
                features,
                labels,
                text_features,
                device,
                chunk_size=cfg.evaluation.chunk_size,
                topk=compute_topk,
            )
            domain_prompt_metrics[domain_key][prompt_template] = metrics
            per_sample_correct[domain_key][prompt_template] = correct
            accuracy_matrix[domain_key].append(metrics["top1_accuracy"])

        LOGGER.info(
            "Domain %s processed with %d samples.",
            domain_key,
            features.size(0),
        )

    domain_keys = list(domain_features.keys())
    if len(domain_keys) != 2:
        LOGGER.warning(
            "Analysis designed for two domains; received %d domains.", len(domain_keys)
        )

    feature_metrics = compute_feature_distances(
        domain_features[domain_keys[0]],
        domain_features[domain_keys[1]],
        cfg.features,
    )

    prompt_sensitivity = compute_prompt_sensitivity(accuracy_matrix)
    domain_sensitivity = compute_domain_sensitivity(
        prompt_templates,
        accuracy_matrix,
        domain_keys,
    )

    anova_results = None
    if cfg.anova.enabled:
        try:
            anova_results = run_two_way_anova(
                per_sample_correct,
                epsilon=cfg.anova.epsilon,
                clip_to_min=cfg.anova.clip_to_min_samples,
            )
        except ValueError as err:
            warnings.warn(f"ANOVA skipped: {err}")

    report = {
        "domains": domain_keys,
        "prompts": prompt_templates,
        "feature_metrics": feature_metrics,
        "accuracy": {
            domain: {
                prompt: {
                    key: float(value)
                    for key, value in domain_prompt_metrics[domain][prompt].items()
                }
                for prompt in prompt_templates
            }
            for domain in domain_keys
        },
        "prompt_sensitivity": prompt_sensitivity,
        "domain_mean_accuracy": {
            domain: float(statistics.mean(values))
            for domain, values in accuracy_matrix.items()
        },
        "domain_sensitivity": domain_sensitivity,
        "domain_sensitivity_mean": float(statistics.mean(domain_sensitivity.values())),
        "accuracy_matrix": {
            domain: [float(x) for x in values]
            for domain, values in accuracy_matrix.items()
        },
        "anova": anova_results,
    }

    sanitized_report = sanitize(report)

    print(json.dumps(sanitized_report, indent=2 if cfg.output.pretty_print else None))

    if cfg.output.save_json:
        out_path = os.path.abspath(cfg.output.json_path)
        ensure_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(
                sanitized_report,
                handle,
                indent=2 if cfg.output.pretty_print else None,
            )
        LOGGER.info("Saved report to %s", out_path)


if __name__ == "__main__":
    main()
