import torch
from torchvision.datasets import ImageFolder
from datasets import Dataset, Features, ClassLabel, Image as HfImage


def make_dataset_dict(path: str, split_name: str = "train"):
    """
    Load images from a folder path using torchvision ImageFolder,
    convert to a HuggingFace Dataset, and return a dict {split_name: dataset}.

    Args:
        path (str): Path to the folder containing class subfolders.
        split_name (str): Name to assign to the resulting split key.

    Returns:
        dict: A mapping from split_name to a HuggingFace Dataset.

    >>> dataset_dict = make_dataset_dict("path/to/folder")
    >>> from datasets import DatasetDict
    >>> dataset_dict = DatasetDict(datasets_dict)
    >>> dataset_dict

    """
    # Use ImageFolder to scan subfolders as classes
    torch_ds = ImageFolder(root=path)

    # Extract image file paths and corresponding labels
    image_paths, labels = zip(*torch_ds.samples)

    # Define HF Dataset features: Image + ClassLabel
    features = Features(
        {"image": HfImage(), "label": ClassLabel(names=torch_ds.classes)}
    )

    # Create a HF Dataset from the collected paths and labels
    hf_ds = Dataset.from_dict(
        {"image": list(image_paths), "label": list(labels)}, features=features
    )

    return {split_name: hf_ds}


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


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def zero_shot_collate_fn(examples):
    images = [example["image"] for example in examples]
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return {
        "images": images,
        "labels": labels,
    }
