import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_sample_prompt_encoder(
    prompt_cfg,
    clip_tokenizer,
    dataset,
):
    labels = dataset.features["label"]
    prompt_template = prompt_cfg.prompt_template

    def _tokenize_labels(example_batch):
        prompt_with_label = [
            prompt_template.format(labels.int2str(label))
            for label in example_batch["label"]
        ]
        text_inputs = clip_tokenizer(
            prompt_with_label, return_tensors="pt", padding=True
        )
        example_batch["input_ids"] = text_inputs.input_ids
        example_batch["attention_mask"] = text_inputs.attention_mask
        return example_batch

    return _tokenize_labels


def build_class_prompt_encoder(zero_shot_dataset, prompt_template, clip_tokenizer):
    def _tokenize_class_captions(example_batch):
        prompt_with_label = [
            prompt_template.format(label)
            for label in zero_shot_dataset.features["label"].names
        ]
        text_inputs = clip_tokenizer(
            prompt_with_label, return_tensors="pt", padding=True
        )
        example_batch["input_ids"] = text_inputs.input_ids
        example_batch["attention_mask"] = text_inputs.attention_mask

    return _tokenize_class_captions
