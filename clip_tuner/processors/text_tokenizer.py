import torch


def get_train_tokenize_fn(dataset, prompt_template, clip_tokenizer):
    def _tokenize_labels(example_batch):
        # 获取
        prompt_with_label = [
            prompt_template.format(dataset.features["label"].int2str(label))
            for label in example_batch["label"]
        ]
        text_inputs = clip_tokenizer(
            prompt_with_label, return_tensors="pt", padding=True
        )
        example_batch["input_ids"] = text_inputs.input_ids
        example_batch["attention_mask"] = text_inputs.attention_mask
        return example_batch

    return _tokenize_labels


def get_zero_shot_tokenize_fn(zero_shot_dataset, prompt_template, clip_tokenizer):
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
