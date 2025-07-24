import sys
import os
from copy import deepcopy
from datasets import load_dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:1"

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPProcessor,
    TrainingArguments,
)
from datasets import load_dataset
import torch

from clip_tuner.data import (
    DataManager,
    get_train_transforms,
    get_val_transforms,
    get_train_tokenize_fn,
    collate_fn,
    zero_shot_collate_fn,
)
from clip_tuner.trainers import BaseTrainer
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
clip_processor = CLIPProcessor.from_pretrained(model_info.path)


data_manager = DataManager(logger=None)  # build data manager without logger
for dataset in data_manager.list_datasets():
    print(dataset)

source_info = data_manager.get_dataset("office-31", "webcam")
target_info = data_manager.get_dataset("office-31", "amazon")

source_dataset = load_dataset(source_info.data_dir)["train"]
target_dataset = load_dataset(target_info.data_dir)["train"]
zero_shot_dataset = deepcopy(target_dataset)

source_dataset.set_transform(
    get_train_transforms(
        image_size=(
            clip_processor.image_processor.crop_size["height"],
            clip_processor.image_processor.crop_size["width"],
        ),
        image_mean=clip_processor.image_processor.image_mean,
        image_std=clip_processor.image_processor.image_std,
    )
)
target_dataset.set_transform(
    get_val_transforms(
        image_size=(
            clip_processor.image_processor.crop_size["height"],
            clip_processor.image_processor.crop_size["width"],
        ),
        image_mean=clip_processor.image_processor.image_mean,
        image_std=clip_processor.image_processor.image_std,
    )
)


model = add_learnable_prompts_to_clip_text_model(clip_model=clip_model)


# 冻住除可训练嵌入参数外的所有参数
for name, param in clip_model.named_parameters():
    param.requires_grad_(False)
    if "learnable_embeddings" in name:
        param.requires_grad = True

# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")


prompt_template = "a photo of a {}."

label2prompt = get_train_tokenize_fn(source_dataset, prompt_template, clip_tokenizer)

source_dataset = source_dataset.map(
    function=label2prompt, batched=True, desc="Running tokenizer on source dataset"
)

# target_dataset = target_dataset.map(
#     label2prompt, batched=True, desc="Running tokenizer on source dataset"
# )

training_args = TrainingArguments(
    output_dir="./output",  # 必须指定（假设输出目录为当前目录下的output）
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    warmup_steps=0,
    weight_decay=0.1,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    remove_unused_columns=False,
    num_train_epochs=2,
    dataloader_num_workers=8,
    eval_strategy="no",  # 完全禁用评估
    save_safetensors=False,
)

trainer = BaseTrainer(
    model=model,
    args=training_args,
    train_dataset=source_dataset if training_args.do_train else None,
    # eval_dataset=target_dataset if training_args.do_eval else None,
    data_collator=collate_fn,
)

# def zero_shot_evaluate(
#     self,
#     zero_shot_eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
#     processor=None,
#     data_collator=None,
#     batch_size=None,
#     compute_per_class_metrics=False,
#     metric_key_prefix: str = "zero_shot_eval",

train_result = trainer.train()
zero_shot_result = trainer.zero_shot_evaluate(
    zero_shot_dataset,
    data_collator=zero_shot_collate_fn,
    processor=clip_processor,
)
# eval_result = trainer.evaluate()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_metrics("zero-shot evaluate", zero_shot_result)
trainer.save_state()
