import time
import json
from typing import Dict, Optional, Union
import math
from tqdm import tqdm

import numpy as np

import torch
from torch import autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.nn.functional as F

from transformers import Trainer, logging
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    seed_worker,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    EvalLoopContainer,
    nested_detach,
)


from .utils import compute_accuracy, compute_per_class_accuracy

logger = logging.get_logger(__name__)


def resolve(x, default):
    return x if x is not None else default


class BaseTrainer(Trainer):

    def __init__(
        self,
        zero_shot_dataset=None,
        zero_shot_collator=None,
        clip_tokenizer=None,
        prompt_template="a photo of a {}",
        *args,
        **kwargs,
    ):
        self.zero_shot_dataset = zero_shot_dataset
        self.zero_shot_collator = zero_shot_collator
        self.clip_tokenizer = clip_tokenizer
        self.prompt_template = prompt_template

        super().__init__(*args, **kwargs)

    def get_class_text_inputs(self, dataset, clip_tokenizer, prompt_template):
        dataset = dataset if (dataset is not None) else self.dataset
        clip_tokenizer = (
            clip_tokenizer if (clip_tokenizer is not None) else self.clip_tokenizer
        )
        prompt_template = (
            prompt_template if (prompt_template is not None) else self.prompt_template
        )
        class_names = dataset.features["label"].names
        prompt_with_label = [prompt_template.format(label) for label in class_names]
        text_inputs = clip_tokenizer(
            prompt_with_label,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs

    def get_zero_shot_dataloader(
        self,
        dataset,
        data_collator,
        batch_size=None,
    ):

        batch_size = (
            self.args.per_device_eval_batch_size if batch_size is None else batch_size
        )
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(dataset)
            dataloader_params["drop_last"] = False
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def zero_shot_evaluate(
        self,
        zero_shot_dataset: Optional[Union[Dataset, Dict[str, Dataset]]],
        zero_shot_collator=None,
        clip_tokenizer=None,
        prompt_template=None,
        batch_size=None,
        compute_per_class_metrics=False,
        metric_key_prefix: str = "zero_shot_eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets

        override = zero_shot_dataset is not None
        zero_shot_dataset = zero_shot_dataset if override else self.zero_shot_dataset
        override = zero_shot_collator is not None
        zero_shot_collator = zero_shot_collator if override else self.zero_shot_collator

        if isinstance(zero_shot_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in zero_shot_dataset.items():
                dataset_metrics = self.zero_shot_evaluate(
                    zero_shot_dataset=(
                        _eval_dataset if override else eval_dataset_name
                    ),
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        text_inputs = self.get_class_text_inputs(
            zero_shot_dataset, clip_tokenizer, prompt_template
        )

        dataloader = self.get_zero_shot_dataloader(
            zero_shot_dataset,
            zero_shot_collator,
            batch_size,
        )

        start_time = time.time()

        output = self.zero_shot_loop(
            dataloader,
            text_inputs,
            compute_per_class_metrics=compute_per_class_metrics,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        logger.info(output.metrics)
        print("ZERO_SHOT_METRICS_START")
        print(json.dumps(output.metrics))  # for bash script to capture
        print("ZERO_SHOT_METRICS_END")

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def zero_shot_loop(
        self,
        dataloader,
        text_inputs,
        compute_per_class_metrics=False,
        metric_key_prefix="zero_shot_eval",
        description="Zero-shot evaluation",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False)
        class_names = dataloader.dataset.features["label"].names
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            if not self.is_in_train:
                if self.args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=self.args.device)
                elif self.args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=self.args.device)

        # Initialize containers
        all_preds = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_labels = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )

        observed_num_examples = 0

        self.control = self.callback_handler.on_prediction_step(
            self.args, self.state, self.control
        )

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.debug(f"  Num examples = {self.num_examples(dataloader)}")
            num_samples = len(dataloader.dataset)
        dataloader_iter = tqdm(
            enumerate(dataloader),
            desc="Zero-Shot Inference",
            total=len(dataloader),
        )

        model.eval()
        with torch.inference_mode(), autocast("cuda"):
            text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}
            text_features = model.get_text_features(**text_inputs).detach()

        for step, image_label_inputs in dataloader_iter:
            # Update the observed num examples
            observed_batch_size = find_batch_size(image_label_inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
            losses, logits_per_image, labels = self.zero_shot_step(
                model,
                image_label_inputs,
                text_features=text_features,
            )

            all_preds.add(logits_per_image)
            all_labels.add(labels)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                del losses, logits_per_image, labels, image_label_inputs
                torch.cuda.empty_cache()
            # break

        if not has_length(dataloader):
            num_samples = observed_num_examples

        # Gather all remaining tensors and put them back on the CPU
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()

        # Metrics!
        if all_preds is not None and all_labels is not None:
            eval_pred = EvalPrediction(predictions=all_preds, label_ids=all_labels)

            metrics = (
                compute_accuracy(eval_pred)
                if not compute_per_class_metrics
                else compute_per_class_metrics(eval_pred, class_names)
            )
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def zero_shot_step(
        self, model, image_label_inputs, text_inputs=None, text_features=None
    ):

        labels = image_label_inputs["labels"]
        pixel_values = image_label_inputs["pixel_values"].to(model.device)
        if text_features is None:
            # If no text features are provided, we need to tokenize the text inputs
            text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}
        model.eval()
        with torch.inference_mode(), autocast("cuda"):
            # 1. Get image features
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            # 2. Get text features if not provided
            if text_features is None:
                text_features = model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, dim=-1)

            # 3. Compute similarity (dot product)
            logits_per_image = image_features @ text_features.T

            loss = None
            if labels is not None:
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits_per_image, labels).detach()
        return (loss, logits_per_image, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # eval_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        zero_shot_metrics = self.zero_shot_evaluate(
            eval_dataset,
            zero_shot_collator=self.zero_shot_collator,
            prompt_template=self.prompt_template,
            clip_tokenizer=self.clip_tokenizer,
        )

        # eval_metrics.update(zero_shot_metrics)
        eval_metrics = zero_shot_metrics
        return eval_metrics
