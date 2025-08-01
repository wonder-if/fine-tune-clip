import torch


def train_collator(examples):
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


def zero_shot_collator(examples):
    images = [example["image"] for example in examples]
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return {
        "images": images,
        "labels": labels,
    }
