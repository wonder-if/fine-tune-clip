import time
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

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


class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

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

    def zero_shot_step(self, model, processor, inputs, candidates):

        labels = inputs["labels"] if "labels" in inputs.keys() else None
        with torch.no_grad():
            processed_inputs = processor(
                images=inputs["images"],
                text=candidates,
                return_tensors="pt",
                padding=True,
            )

            processed_inputs = processed_inputs.to(model.device)
            output = model(**processed_inputs)
        loss = output.loss.mean().detach() if output.loss is not None else None
        logits_per_image = output.logits_per_image
        return (loss, logits_per_image, labels)

    def zero_shot_loop(
        self,
        dataloader,
        class_names,
        processor,
        compute_per_class_metrics=False,
        metric_key_prefix="zero_shot_eval",
        description="Zero-shot evaluation",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False)
        candidates = [
            f"a photo of a {label_name.replace('_', ' ')}" for label_name in class_names
        ]
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

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
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")

        for step, image_label_inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(image_label_inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            losses, logits_per_image, labels = self.zero_shot_step(
                model, processor, image_label_inputs, candidates
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

        if has_length(dataloader):
            num_samples = len(dataloader.dataset)
        else:
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

    def zero_shot_evaluate(
        self,
        zero_shot_eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]],
        data_collator,
        processor,
        batch_size=None,
        compute_per_class_metrics=False,
        metric_key_prefix: str = "zero_shot_eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = zero_shot_eval_dataset is not None
        if isinstance(zero_shot_eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in zero_shot_eval_dataset.items():
                dataset_metrics = self.zero_shot_evaluate(
                    zero_shot_eval_dataset=(
                        _eval_dataset if override else eval_dataset_name
                    ),
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        batch_size = batch_size if batch_size is not None else self.args.eval_batch_size

        dataloader = self.get_zero_shot_dataloader(
            zero_shot_eval_dataset,
            data_collator,
            batch_size,
        )

        start_time = time.time()
        class_names = zero_shot_eval_dataset.features["label"].names

        output = self.zero_shot_loop(
            dataloader,
            class_names=class_names,
            processor=processor,
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

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
