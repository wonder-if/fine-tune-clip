from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
    RandomGrayscale,
    ToTensor,
    Resize,
)


def get_train_transforms(
    image_size=(224, 224),
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    blur=False,
):
    if blur:
        _train_transforms = Compose(
            [
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=0.4,  # 亮度
                    contrast=0.4,  # 对比度
                    saturation=0.4,  # 饱和度
                    hue=0.2,  # 色调,
                ),
                GaussianBlur(kernel_size=5),
                RandomGrayscale(p=0.2),
                ToTensor(),
                Normalize(image_mean, image_std),
            ]
        )
    else:
        _train_transforms = Compose(
            [
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2,
                ),
                ToTensor(),
                Normalize(image_mean, image_std),
            ]
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        # 对一个批次的数据应用_train_transforms
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]

        del example_batch["image"]
        return example_batch

    return train_transforms


def get_val_transforms(
    image_size=224,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
):
    _val_transforms = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(image_mean, image_std),
        ]
    )

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        del example_batch["image"]
        return example_batch

    return val_transforms
