import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPImageProcessor,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch

from clip_tuner.data import DataManager, get_train_transforms, get_val_transforms
from clip_tuner.utils import Logger
from clip_tuner.models import (
    ModelManager,
    add_learnable_prompts_to_clip_text_model,
)


logger = Logger(log_dir="../logs/", log_file="train_demo.log", file_mode="w")
print(f'created logger at "{logger.log_path}"')


model_manager = ModelManager()
for model_info in model_manager.list_models():
    print(model_info)
model_info = model_manager.get_model("clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained(model_info.path)
clip_tokenizer = CLIPTokenizer.from_pretrained(model_info.path)
clip_image_processor = CLIPImageProcessor.from_pretrained(model_info.path)

clip_model, clip_tokenizer, clip_image_processor


data_manager = DataManager(logger=None)  # build data manager without logger
for dataset in data_manager.list_datasets():
    print(dataset)

source_info = data_manager.get_dataset("office-31", "webcam")
target_info = data_manager.get_dataset("office-31", "amazon")

source_dataset = load_dataset(source_info.data_dir)["train"]
target_dataset = load_dataset(target_info.data_dir)["train"]


source_dataset.set_transform(
    get_train_transforms(
        image_size=(
            clip_image_processor.crop_size["height"],
            clip_image_processor.crop_size["width"],
        ),
        image_mean=clip_image_processor.image_mean,
        image_std=clip_image_processor.image_std,
    )
)
target_dataset.set_transform(
    get_val_transforms(
        image_size=(
            clip_image_processor.crop_size["height"],
            clip_image_processor.crop_size["width"],
        ),
        image_mean=clip_image_processor.image_mean,
        image_std=clip_image_processor.image_std,
    )
)

source_dataset, source_dataset.features["label"].names, target_dataset


model = add_learnable_prompts_to_clip_text_model(clip_model=clip_model)


# 冻住除可训练嵌入参数外的所有参数
# for name, param in clip_model.named_parameters():
#     param.requires_grad_(False)
#     if "learnable_embeddings" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")


prompt_template = "a photo of a {}."


def label2prompt(example_batch):
    # 获取
    prompt_with_label = [
        prompt_template.format(source_dataset.features["label"].int2str(label))
        for label in example_batch["label"]
    ]
    text_inputs = clip_tokenizer(prompt_with_label, return_tensors="pt", padding=True)
    example_batch["input_ids"] = text_inputs.input_ids
    example_batch["attention_mask"] = text_inputs.attention_mask
    return example_batch


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


source_dataset = source_dataset.map(
    function=label2prompt, batched=True, desc="Running tokenizer on source dataset"
)

target_dataset = target_dataset.map(
    label2prompt, batched=True, desc="Running tokenizer on source dataset"
)

training_args = TrainingArguments()

training_args.per_device_train_batch_size = 64
training_args.per_device_eval_batch_size = 64
training_args.learning_rate = 5e-5
training_args.warmup_steps = 0
training_args.weight_decay = 0.1
training_args.overwrite_output_dir = True
training_args.do_train = True
training_args.do_eval = True
training_args.remove_unused_columns = False
training_args.num_train_epochs = 20

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=source_dataset if training_args.do_train else None,
    eval_dataset=target_dataset if training_args.do_eval else None,
    data_collator=collate_fn,
)


train_result = trainer.train()
# trainer.save_model()
# trainer.log_metrics("train", train_result.metrics)
# trainer.save_metrics("train", train_result.metrics)
# trainer.save_state()
